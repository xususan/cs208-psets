from absl import app
from absl import flags

import numpy as np
import tensorflow as tf


from privacy.analysis import privacy_ledger
from privacy.analysis.rdp_accountant import compute_rdp_from_ledger
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers

# Compatibility with tf 1 and 2 APIs
try:
	GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
	AdamOptimizer = tf.compat.v1.train.AdamOptimizer
	AdagradOptimizer = tf.compat.v1.train.AdagradOptimizer
except:  # pylint: disable=bare-except
	GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

FLAGS = flags.FLAGS
flags.DEFINE_boolean(
	'dp', True, 'If True, train with DP-optimizer. If False, '
	'train with non-private optimizer.')
flags.DEFINE_string(
	'optim', 'sgd', 'which optimizer to use')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
					 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_float('c', .00001, 'L2 regularization constant (C)')
flags.DEFINE_float('dropout', None, 'Dropout probability')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('layers', 1, 'number of layers')
flags.DEFINE_string('cnn_size', 'full', 'size of cnn: full / mini')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer(
	'microbatches', 256, 'Number of microbatches '
	'(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_string('model', None, 'ff, lr, or cnn')
flags.DEFINE_boolean('pca', False, 'if True, use PCA models. if False, use full images.')
flags.DEFINE_boolean('verbose', True, 'if True, info level logging. Else error only ')

num_classes=10

class EpsilonPrintingTrainingHook(tf.estimator.SessionRunHook):
	"""Training hook to print current value of epsilon after an epoch."""

	def __init__(self, ledger):
		"""Initalizes the EpsilonPrintingTrainingHook.

		Args:
			ledger: The privacy ledger.
		"""
		self._samples, self._queries = ledger.get_unformatted_ledger()

	def end(self, session):
		orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
		samples = session.run(self._samples)
		queries = session.run(self._queries)
		formatted_ledger = privacy_ledger.format_ledger(samples, queries)
		rdp = compute_rdp_from_ledger(formatted_ledger, orders)
		eps = get_privacy_spent(orders, rdp, target_delta=1e-5)[0]
		print('For delta=1e-5, the current epsilon is: %.2f' % eps)

def generate_estimator_spec(logits, features, labels, mode):
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
		'probabilities': tf.nn.softmax(logits),
		'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)
	# Calculate loss as a vector (to support microbatches in DP-SGD).
	vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= labels,logits=logits)
	# Define mean of loss across minibatch (for reporting through tf.Estimator).
	scalar_loss = tf.reduce_mean(vector_loss)

	# Configure the training op (for TRAIN mode).
	if mode == tf.estimator.ModeKeys.TRAIN:

		if FLAGS.dp:
			ledger = privacy_ledger.PrivacyLedger(
				population_size=60000,
				selection_probability=(FLAGS.batch_size / 60000))

			# Use DP version of GradientDescentOptimizer. Other optimizers are
			# available in dp_optimizer. Most optimizers inheriting from
			# tf.train.Optimizer should be wrappable in differentially private
			# counterparts by calling dp_optimizer.optimizer_from_args().
			if FLAGS.optim == 'sgd':
				optimizer_func = dp_optimizer.DPGradientDescentGaussianOptimizer
			elif FLAGS.optim == 'adam':
				optimizer_func = dp_optimizer.DPAdamGaussianOptimizer
			elif FLAGS.optim == 'adagrad':
				optimizer_func = dp_optimizer.DPAdagradGaussianOptimizer
			else:
				raise ValueError("optimizer function not supported")



			optimizer = optimizer_func(
				l2_norm_clip=FLAGS.l2_norm_clip,
				noise_multiplier=FLAGS.noise_multiplier,
				num_microbatches=FLAGS.microbatches,
				ledger=ledger,
				learning_rate=FLAGS.learning_rate)
			training_hooks = [
				EpsilonPrintingTrainingHook(ledger)
			]
			opt_loss = vector_loss
		else:
			if FLAGS.optim == 'sgd':
				optimizer_func = GradientDescentOptimizer
			elif FLAGS.optim == 'adam':
				optimizer_func = AdamOptimizer
			elif FLAGS.optim == 'adagrad':
				optimizer_func = AdagradOptimizer
			else:
				raise ValueError("optimizer function not supported")
			optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
			training_hooks = []
			opt_loss = scalar_loss
		global_step = tf.train.get_global_step()
		train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
		# In the following, we pass the mean of the loss (scalar_loss) rather than
		# the vector_loss because tf.estimator requires a scalar loss. This is only
		# used for evaluation and debugging by tf.estimator. The actual loss being
		# minimized is opt_loss defined above and passed to optimizer.minimize().
		return tf.estimator.EstimatorSpec(mode=mode,
											loss=scalar_loss,
											train_op=train_op,
											training_hooks=training_hooks)

	# Add evaluation metrics (for EVAL mode).
	elif mode == tf.estimator.ModeKeys.EVAL:
		eval_metric_ops = {
			'accuracy':
				tf.metrics.accuracy(
					labels=labels,
					predictions=tf.argmax(input=logits, axis=1)),
			'crossentropy':
				tf.metrics.mean(
					scalar_loss
					)
		}

		return tf.estimator.EstimatorSpec(mode=mode,
											loss=scalar_loss,
											eval_metric_ops=eval_metric_ops)


def lr_nonpca_model_fn(features, labels, mode):
	""" Model function for logistic regression."""


	# C = .00001
	C = FLAGS.c

	# Define CNN architecture using tf.keras.layers. layers.Flatten(input_shape=(32, 32, 3)))
	input_layer = tf.reshape(features['x'], [-1, 32, 32, 3])
	# model = models.Sequential()
	y = layers.Flatten().apply(input_layer)

	logits = layers.Dense(
		num_classes,
		kernel_regularizer=tf.keras.regularizers.l2(C)).apply(y)

	return generate_estimator_spec(logits, features, labels, mode)

def lr_model_fn(features, labels, mode):
	""" Model function for logistic regression."""
	"""Model function for a feed forward network."""

	C = .00001

	def lr_model(x):
		# Define CNN architecture using tf.keras.layers.
		input_layer = tf.reshape(x, [-1, 50])
		# model = models.Sequential()
		y = layers.Dense(
			num_classes, kernel_regularizer=tf.keras.regularizers.l2(C)).apply(input_layer)
		return y

	return model_function_from_model(lr_model, features, labels, mode)



def ff_model_fn(features, labels, mode):
	"""Model function for a feed forward network."""
	C = FLAGS.c

	dropout_p = FLAGS.dropout

	# Define FF architecture using tf.keras.layers.
	# input_layer = tf.reshape(x, [-1, 50])

	input_layer = tf.reshape(features['x'], [-1, 32, 32, 3])
	y = layers.Flatten().apply(input_layer)


	for _ in range(FLAGS.layers):
		y = layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(C)).apply(y)

		if not FLAGS.model_dir == 'ff_nondp':
			y = layers.Dropout(dropout_p).apply(y)


	logits= layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(C)).apply(y)


	return generate_estimator_spec(logits, features, labels, mode)

	# return model_function_from_model(ff_model, features, labels, mode)


def cnn_model_fn(features, labels, mode):
	"""Model function for a CNN."""

	def cnn_model_4dropout(x):
		# Define CNN architecture using tf.keras.layers.
		dropout_p = FLAGS.dropout
		y = tf.reshape(x, [-1, 32, 32, 3])
		y = layers.Conv2D(96, (3, 3), padding='same', activation='relu').apply(y)
		y = layers.Dropout(dropout_p).apply(y)
		y = layers.Conv2D(filters=192, kernel_size=(3,3), strides=2, activation='relu').apply(y)
		y = layers.Dropout(dropout_p).apply(y)
		y = layers.Conv2D(filters=192, kernel_size=(3,3), strides=2, activation='relu').apply(y)
		y = layers.Dropout(dropout_p).apply(y)
		y = layers.Conv2D(filters=192, kernel_size=(3,3), strides=2, activation='relu').apply(y)
		y = layers.Dropout(dropout_p).apply(y)

		y = layers.Flatten().apply(y)
		y = layers.BatchNormalization().apply(y)
		y = layers.Dense(256, activation='relu').apply(y)
		logits = layers.Dense(num_classes).apply(y)
		return logits

	def cnn_mini(x):
		# Define CNN architecture using tf.keras.layers.
		dropout_p = FLAGS.dropout
		y = tf.reshape(x, [-1, 32, 32, 3])
		y = layers.Conv2D(filters=36, kernel_size=(3, 3), padding='same', activation='relu').apply(y)
		y = layers.Dropout(dropout_p).apply(y)
		y = layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, activation='relu').apply(y)
		y = layers.Dropout(dropout_p).apply(y)
		y = layers.Flatten().apply(y)
		# y = layers.BatchNormalization().apply(y)
		# y = layers.Dense(256, activation='relu').apply(y)
		logits = layers.Dense(num_classes).apply(y)
		return logits

	def cnn_model(x):
		# Define CNN architecture using tf.keras.layers.
		y = tf.reshape(x, [-1, 32, 32, 3])
		y = layers.Conv2D(96, (3, 3), padding='same').apply(y)
		y = layers.Activation('relu').apply(y)
		y = layers.Conv2D(filters=192, kernel_size=(3,3), strides=2).apply(y)
		y = layers.Activation('relu').apply(y)
		y = layers.Dropout(0.25).apply(y)
		y = layers.Conv2D(filters=192, kernel_size=(3,3), strides=2).apply(y)

		y = layers.Activation('relu').apply(y)
		y = layers.Conv2D(filters=192, kernel_size=(3,3), strides=2).apply(y)
		y = layers.Activation('relu').apply(y)
		y = layers.Dropout(0.5).apply(y)

		y = layers.Flatten().apply(y)
		y = layers.BatchNormalization().apply(y)
		y = layers.Dense(256).apply(y)
		y = layers.Activation('relu').apply(y)
		logits = layers.Dense(num_classes).apply(y)
		return logits

	if FLAGS.cnn_size == 'mini':
		logits = cnn_mini(features['x'])
	else:
		if FLAGS.model_dir == 'cnn_nondp':
			logits = cnn_model(features['x'])
		else:
			logits = cnn_model_4dropout(features['x'])
	return generate_estimator_spec(logits, features, labels, mode)


def load_cifar_pca():
	print('loaded cifar pca')
	xs = np.load('data/cifar_100_features.p',allow_pickle=True)
	ys = np.load('data/cifar_100_labels.p', allow_pickle=True)

	# train_data, test_data, train_labels, test_labels = train_test_split(xs, ys, train_size=10000, random_state=31415)

	train_data = np.array(train_data, dtype=np.float32) 
	test_data = np.array(test_data, dtype=np.float32)

	train_labels = np.array(train_labels, dtype=np.int32).reshape(-1,)
	test_labels = np.array(test_labels, dtype=np.int32).reshape(-1,)
	held_data = train_data[10000:]
	train_data = train_data[:10000]

	held_lables = train_labels[10000:]
	train_labels = train_labels[:10000]

	return train_data, train_labels, test_data, test_labels

def count_params():
    "print number of trainable variables"
    size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size(v) for v in tf.trainable_variables())
    print("Model size: %d" % (n,))


def load_cifar():
	"""Loads MNIST and preprocesses to combine training and validation data."""
	if num_classes == 100:
		train, test = tf.keras.datasets.cifar100.load_data()
	else:
		train, test = tf.keras.datasets.cifar10.load_data()

	train_data, train_labels = train
	test_data, test_labels = test

	train_data = np.array(train_data, dtype=np.float32) / 255.0
	test_data = np.array(test_data, dtype=np.float32) / 255.0

	train_labels = np.array(train_labels, dtype=np.int32).reshape(-1,)
	test_labels = np.array(test_labels, dtype=np.int32).reshape(-1,)

	held_data = train_data[10000:20000]
	train_data = train_data[:10000]

	held_lables = train_labels[10000:20000]
	train_labels = train_labels[:10000]

	assert train_data.min() == 0.
	assert train_data.max() == 1.
	assert test_data.min() == 0.
	assert test_data.max() == 1.
	assert train_labels.ndim == 1
	assert test_labels.ndim == 1

	return train_data, train_labels, test_data, test_labels, held_lables, held_data

def attack(model, positive_test_set, negative_test_set, positive_ys, negative_ys):

	all_losses = []
	for input_func, ys in (zip([positive_test_set, negative_test_set], [positive_ys, negative_ys])):
		ys_onehot = tf.keras.utils.to_categorical(ys, num_classes=10)
		predictions = model.predict(input_fn=input_func)

		logits = np.array(list(x['probabilities'] for x in predictions))
		losses = -1*np.sum(ys_onehot * np.log(logits), axis=1)
		all_losses.append(losses)

	for threshold in np.linspace(.1, 5.1, 11):
		tp = np.sum(np.array(all_losses[0]) < threshold)
		fp = np.sum(np.array(all_losses[1]) < threshold)
		recall = np.mean((np.array(all_losses[0]) < threshold))
		precision = tp / (tp + fp)
		print("%f\t%f\t%f" % (threshold, precision, recall))
	return



def attack2(model, input_func, ys, training_error):
	ys_onehot = tf.keras.utils.to_categorical(ys, num_classes=10)
	predictions = model.predict(input_fn=input_func)

	logits = np.array(list(x['probabilities'] for x in predictions))
	#if use_logits:
	#	probs = softmax(logits, axis=1)
	#else:
	#	probs = logits
	losses = -1*np.sum(ys_onehot * np.log(logits), axis=1)
	return np.mean(np.array(losses) < training_error)
	