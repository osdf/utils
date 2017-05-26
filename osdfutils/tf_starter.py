import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

with tf.device("/gpu:0"):
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

logits = (tf.matmul(x, W) + b)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

updt = tf.train.AdamOptimizer()
gvs = updt.compute_gradients(loss)
optim = updt.apply_gradients(gvs)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    _, _loss = sess.run([optim, loss], feed_dict={x: batch[0], y_: batch[1]})
    print _loss
