import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

# training data
tx = np.linspace(0,50,50) + np.random.uniform(-4,4,50)
# xshape = (50,3)
# tx = np.tile(np.linspace(0,50,50), xshape) + np.random.randn(xshape)
ty = np.linspace(0,50,50) + np.random.uniform(-4,4,50)

# fig = plt.figure(figsize=(8,8))
# plt.scatter(tx, ty)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Training Data')
# plt.show()

# placeholders for data in the graph
X = tf.placeholder("float")
y = tf.placeholder("float")

# Variables that are learnt
W = tf.Variable(np.random.randn(), name='W')
b = tf.Variable(np.random.randn(), name='b')

#
y_pred = tf.add(tf.multiply(X, W), b)
cost = tf.reduce_sum(tf.pow(y_pred - y, 2)) / (1. * len(tx))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=.9).minimize(cost)

init = tf.global_variables_initializer()


fig = plt.figure(figsize=(8,8))
plt.scatter(tx, ty, color='r', label='Original data')
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(40):
		for _x, _y in zip(tx, ty):
			sess.run(optimizer, feed_dict={X: _x, y: _y})
			# if (epoch % 2) == 0:
			# 	c = sess.run(cost, feed_dict={X: tx, y: ty})
			# 	print("Epoch : {}, cost: {}, W: {}, b: {}".format(epoch, c, sess.run(W), sess.run(b)))
		weight = sess.run(W)
		bias = sess.run(b)
		predictions = weight * tx + bias
		plt.plot(tx, predictions, color='b')

	training_cost = sess.run(cost, feed_dict={X: tx, y: ty})
	weight = sess.run(W)
	bias = sess.run(b)

print("Training cost: {}, Weight: {}, Bias: {}".format(training_cost,weight,bias))
plt.plot(tx, predictions, color='b', label='Fitted line')
plt.legend()
plt.show()
