"""
This is a 1 layer neural network trained using Tensorflow.
It closely follows the Tensorflow beginner tutorials.

It was used for digit classification using the MNIST dataset
ACCURACY: ~95%
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#load the dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#First hidden layer
x = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.truncated_normal([784, 30], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[30]))
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

#output layer
W = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))
y = tf.nn.softmax(tf.matmul(y1, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

#use stochastic gradient descent + cross entropy with constant learning rate
cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) + tf.reduce_sum(W1)/(784*30) + tf.reduce_sum(W)/(10*30) 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#train for multiple epochs
for i in range(2000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

	if i%100 == 0:
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

# weights = sess.run(W)
# for weight in weights:
# 	print(weight)
# 	
print("VALIDATION")
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x:mnist.validation.images, y_:mnist.validation.labels}))
sess.close()
