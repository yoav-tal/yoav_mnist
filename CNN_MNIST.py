""" Solution for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 20
n_train = 60000
n_test = 10000
MNIST = True

# convolution layer parameters
n_filters = 5
filter_size = 5

# Step 1: Get data
train_data, test_data = utils.get_mnist_dataset(batch_size, MNIST)

iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()
img = tf.cast(img, tf.float32)
img = tf.reshape(img,[-1,28,28,1])

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 2: Set layers structure

H1 = utils.conv_layer(img, n_filters, filter_size)
H1_relu = tf.nn.relu(H1)
H1_pool = tf.nn.max_pool(H1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding = "SAME")
feature_dim = H1_pool.shape[1] * H1_pool.shape[2] * H1_pool.shape[3]
logits = tf.layers.dense(tf.reshape(H1_pool,[-1,feature_dim]),10,kernel_initializer=tf.random_normal_initializer(0, 0.01),name = 'logits')
#H2 = tf.layers.dense(tf.sigmoid(H1),100,kernel_initializer=tf.random_normal_initializer(0, 0.01),name = 'H2')


# Step 3: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss') # computes the mean over all the examples in the batch

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

#writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:

    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        j=0
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                j=j+1
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))
#writer.close()