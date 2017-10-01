import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
import math

t0 = time.time()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
ntrain_batches = 2000
ntest_batches = 200
batch_size = 50

img = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])
ans = tf.placeholder(tf.float32, [batch_size, 10])

flts_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
conv_out_1 = tf.nn.conv2d(img, flts_1, strides=[1, 2, 2, 1], padding="SAME")
conv_out_1 = tf.nn.relu(conv_out_1)
pool_out_1 = tf.nn.max_pool(conv_out_1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding="SAME")

flts_2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
conv_out_2 = tf.nn.conv2d(pool_out_1, flts_2, strides=[1, 2, 2, 1], padding="SAME")
conv_out_2 = tf.nn.relu(conv_out_2)
pool_out_2 = tf.nn.max_pool(conv_out_2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding="SAME")

nfeatures = int(math.ceil(28/2/2/2/2) ** 2 * 64)
nlabels = 10
pool_out_2 = tf.reshape(pool_out_2, [batch_size, nfeatures])
U = tf.Variable(tf.random_normal([nfeatures, nfeatures], stddev=.1))
b_U = tf.Variable(tf.random_normal([nfeatures], stddev=.1))
V = tf.Variable(tf.random_normal([nfeatures, nlabels], stddev=.1))
b_V = tf.Variable(tf.random_normal([nlabels], stddev=.1))

L1_output = tf.nn.relu(tf.matmul(pool_out_2, U) + b_U)
probs = tf.nn.softmax(tf.matmul(L1_output, V) + b_V)
ent_loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(probs), reduction_indices=[1]))

train = tf.train.AdamOptimizer(1e-4).minimize(ent_loss)
ncorrect = tf.equal(tf.argmax(probs, 1), tf.argmax(ans, 1))
accuracy = tf.reduce_mean(tf.cast(ncorrect, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(ntrain_batches):
	curr_imgs, curr_ans = mnist.train.next_batch(batch_size)
	curr_imgs = np.reshape(curr_imgs, [batch_size, 28, 28, 1])
	session.run(train, feed_dict={img: curr_imgs, ans: curr_ans}) 

sum_acc = 0
for i in range(ntest_batches):
	curr_imgs, curr_ans = mnist.train.next_batch(batch_size)
	curr_imgs = np.reshape(curr_imgs, [batch_size, 28, 28, 1])
	sum_acc += session.run(accuracy, feed_dict={img: curr_imgs, ans: curr_ans})

print("Test accuracy: %r" % (sum_acc / ntest_batches))
print("Time elapsed: %r seconds" % (time.time() - t0))