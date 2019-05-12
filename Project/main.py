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
		train_data, train_labels, test_data, test_labels, held_lables, held_data = load_cifar()

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

	# Training loop.
	steps_per_epoch = 60000 // FLAGS.batch_size
	for epoch in range(1, FLAGS.epochs + 1):
	# Train the model for one epoch.
		mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
		eval_results = mnist_classifier.evaluate(input_fn=train_input_fn)
		print('Epoch %d / %d: Train Accuracy %.3f \t Loss %.3f' % (epoch, FLAGS.epochs + 1, eval_results['accuracy'], eval_results['crossentropy']))


		# Evaluate the model and print results
		eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
		test_accuracy = eval_results['accuracy']
		print('Epoch %d / %d: Test Accuracy %.3f \t Loss %.3f' % (epoch, FLAGS.epochs + 1, eval_results['accuracy'], eval_results['crossentropy']))

if __name__ == '__main__':
	app.run(main)
