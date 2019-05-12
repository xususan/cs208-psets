

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from absl import app
from absl import flags

import numpy as np
import tensorflow as tf

from models import *


np.random.seed(31415)
tf.random.set_random_seed(31415)




def main(unused_argv):
	tf.logging.set_verbosity(tf.logging.INFO)
	if FLAGS.dp and FLAGS.batch_size % FLAGS.microbatches != 0:
		raise ValueError('Number of microbatches should divide evenly batch_size')

	# Load training and test data.
	if FLAGS.pca:
		train_data, train_labels, test_data, test_labels = load_cifar_pca()
	else:
		train_data, train_labels, test_data, test_labels, held_labels, held_data = load_cifar()

	print(train_data.shape)

	if FLAGS.model == 'cnn':
		model_function = cnn_model_fn
	elif FLAGS.model == 'ff':
		model_function = ff_model_fn
	elif FLAGS.model == 'lr':
		if FLAGS.pca:
			model_function = lr_model_fn
		else:
			model_function = lr_nonpca_model_fn
			print("NON PCA LOGISTIC")
	else:
		raise ValueError('not supported flags.model')

	# Instantiate the tf.Estimator.
	mnist_classifier = tf.estimator.Estimator(model_fn=model_function,
											model_dir=FLAGS.model_dir)

	# Create tf.Estimator input functions for the training and test data.
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x': train_data},
		y=train_labels,
		batch_size=FLAGS.batch_size,
		num_epochs=FLAGS.epochs,
		shuffle=True)
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x': test_data},
		y=test_labels,
		num_epochs=1,
		shuffle=False)
	held_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x': held_data},
		y=held_labels,
		num_epochs=1,
		shuffle=False)

	predict_training_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x': train_data},
		batch_size=1,
		shuffle=False)


	eval_results = mnist_classifier.evaluate(input_fn=train_input_fn)
	train_accuracy = eval_results['accuracy']
	train_loss = eval_results['crossentropy']
	print('Train accuracy is: %.3f, Train loss is : %.3f' % (train_accuracy, train_loss))

	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	test_acc = eval_results['accuracy']
	test_loss = eval_results['crossentropy']
	print('Test accuracy is: %.3f, test loss is : %.3f' % (test_acc, test_loss))


	tpr = attack2(mnist_classifier, predict_training_input_fn, train_labels, train_loss)
	print("Membership Inference Attack: True positive rate = %f" %(tpr))

	fnr = attack2(mnist_classifier, held_input_fn, held_labels, train_loss)
	print("Membership Inference Attack: True negative rate = %f (held)" %(1 - fnr))

	fnr = attack2(mnist_classifier, eval_input_fn, test_labels, train_loss)
	print("Membership Inference Attack: True negative rate = %f (test)" %(1 - fnr))


if __name__ == '__main__':
	app.run(main)