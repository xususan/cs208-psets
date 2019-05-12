# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training a CNN on CIFAR-100 with differentially private SGD optimizer."""

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
	if FLAGS.verbose:
		tf.logging.set_verbosity(tf.logging.INFO)
	else:
		tf.logging.set_verbosity(tf.logging.ERROR)

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

	checkpoint_config = tf.estimator.RunConfig(
		keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
	)


	# Instantiate the tf.Estimator.
	mnist_classifier = tf.estimator.Estimator(model_fn=model_function,
											model_dir=FLAGS.model_dir,
											config=checkpoint_config)

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

	# Training loop.
	steps_per_epoch = 60000 // FLAGS.batch_size
	for epoch in range(1, FLAGS.epochs + 1):
	# Train the model for one epoch.
		mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
		eval_results = mnist_classifier.evaluate(input_fn=train_input_fn)
		train_loss = eval_results['crossentropy']
		print('=================================================================')
		print('Epoch %d / %d: Train Accuracy %.3f \t Loss %.3f' % (epoch, FLAGS.epochs, eval_results['accuracy'], eval_results['crossentropy']))


		# Evaluate the model and print results
		eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
		test_accuracy = eval_results['accuracy']
		print('\t \t : Test Accuracy %.3f \t Loss %.3f' % ( eval_results['accuracy'], eval_results['crossentropy']))

		tpr = attack2(mnist_classifier, predict_training_input_fn, train_labels, train_loss)
		print("Membership Inference Attack: True positive rate = %f" %(tpr))

		fnr = attack2(mnist_classifier, held_input_fn, held_labels, train_loss)
		print("Membership Inference Attack: True negative rate = %f (held)" %(1 - fnr))

		fnr = attack2(mnist_classifier, eval_input_fn, test_labels, train_loss)
		print("Membership Inference Attack: True negative rate = %f (test)" %(1 - fnr))

if __name__ == '__main__':
	app.run(main)
