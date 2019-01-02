
#
# Following fails due to compatibility issues between Edward and latest Tensorflow
# https://github.com/blei-lab/edward/issues/882
#
import edward as ed
import tensorflow as tf
from edward.models import Binomial, Normal

N = 5000 # individuals
M = 100000 # SNPs
K = 25 # latent dimension

def snp_neural_network(z, w):
	z_tile = tf.tile(tf.reshape(z, [N, 1, K]), [1, M, 1])
	w_tile = tf.tile(tf.reshape(w, [1, M, K]), [N, 1, 1])
	h = tf.concat([z_tile, w_tile], 2)
	h = tf.layers.dense(h, 512, activation=tf.nn.relu)
	h = tf.layers.dense(h, 512, activation=tf.nn.relu)
	h = tf.layers.dense(h, 1, activation=None)
	return tf.reshape(h, [N, M])

def trait_neural_network(z, x):
	eps = np.random.normal(loc=0., scale=1., size=(N, 1))
	h = tf.concat([z, x, eps], 1)
	h = tf.layers.dense(h, 32, activation=tf.nn.relu)
	h = tf.layers.dense(h, 256, activation=tf.nn.relu)
	h = tf.concat([z, h], 1)
	h = tf.layers.dense(h, 1, activation=None)
	return tf.reshape(h, [N])

z = np.random.normal(loc=0., scale=1., size=(N, K))
w = np.random.normal(loc=0., scale=1., size=(M, K))
logits = snp_neural_network(z, w)
print(type(logits))
x = binom(2, 1./(1. + tf.exp(-logits)))

y = trait_neural_network(z, x)
