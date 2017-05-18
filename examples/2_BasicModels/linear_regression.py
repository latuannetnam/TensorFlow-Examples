'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
logs_path = '/tmp/tensorflow_logs/linear'

# Training Data
# train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
#                          7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
# train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
#                          2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
train_X = np.array([np.arange(100)]).T
train_Y = 2 * train_X + 5
n_samples = train_X.size


with tf.name_scope('Input'):
    # tf Graph Input
    X = tf.placeholder("float", name='x')
with tf.name_scope('Label'):
    Y = tf.placeholder("float", name='y')
with tf.name_scope("Weights"):
    # Set model weights
    W = tf.Variable(rng.randn(), name="weights")
    b = tf.Variable(rng.randn(), name="bias")
with tf.name_scope('Model'):
    # Construct a linear model
    pred = tf.add(tf.multiply(X, W), b)
with tf.name_scope('Loss'):
    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
with tf.name_scope('Train'):
    # Gradient descent
    # Note, minimize() knows to modify W and b because Variable objects are
    # trainable=True by default
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # Gradient Descent
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # # Op to calculate every variable gradient
    # grads = tf.gradients(cost, tf.trainable_variables())
    # grads = list(zip(grads, tf.trainable_variables()))
    # # Op to update all variables according to their gradient
    # apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
# Initializing the variables
init = tf.global_variables_initializer()
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create summaries to visualize weights
# for var in tf.trainable_variables():
#     tf.summary.histogram(var.name, var)
# Summarize all gradients
# for grad, var in grads:
#     tf.summary.histogram(var.name + '/gradient', grad)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(
        logs_path, graph=tf.get_default_graph())
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            # _, summary = sess.run(
            #     [apply_grads, merged_summary_op], feed_dict={X: x, Y: y})
            _, summary = sess.run(
                [optimizer, merged_summary_op], feed_dict={X: x, Y: y})
        # _, summary = sess.run(
        #         [apply_grads, merged_summary_op], feed_dict={X: train_X, Y: train_Y})
        # Write logs at every iteration
        summary_writer.add_summary(summary, epoch + 1)

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            # c = sess.run(cost)
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=",
          sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    # print("Testing... (Mean square loss Comparison)")
    # testing_cost = sess.run(
    #     tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
    #     feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    # print("Testing cost=", testing_cost)
    # print("Absolute mean square loss difference:", abs(
    #     training_cost - testing_cost))

    # plt.plot(test_X, test_Y, 'bo', label='Testing data')
    # plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()
