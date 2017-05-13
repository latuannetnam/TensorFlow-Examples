'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
logs_path = '/tmp/tensorflow_logs/linear_polynomial'

# Training Data
train_X = numpy.arange(10)
train_Y = 2 * numpy.multiply(train_X, train_X) + 6 * train_X - 15
n_samples = train_X.shape[0]
print(train_X)
print(train_Y)

with tf.name_scope('Input'):
    # tf Graph Input
    X = tf.placeholder("float", name='x1')
with tf.name_scope('Label'):
    Y = tf.placeholder("float", name='y1')
with tf.name_scope('Weights'):
    # Set model weights
    W1 = tf.Variable(rng.randn(), name="W1")
    W2 = tf.Variable(rng.randn(), name="W2")
with tf.name_scope('Bias'):
    b = tf.Variable(rng.randn(), name="b")
with tf.name_scope('Model'):
    # Construct a linear model
    pred = tf.multiply(X, W1) + tf.multiply(tf.pow(X, 2), W2) + b
with tf.name_scope('Loss'):
    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
with tf.name_scope('SGD'):
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
            _, c, summary = sess.run(
                [optimizer, cost, merged_summary_op], feed_dict={X: x, Y: y})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch + 1)

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W1=", sess.run(W1), "W2=", sess.run(W2), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W1=",
          sess.run(W1), "W2=", sess.run(W2), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W1) * train_X + sess.run(W2) *
             numpy.multiply(train_X, train_X) + sess.run(b), label='Fitted line')
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
