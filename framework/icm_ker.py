from keras import backend
from keras.utils import plot_model
from keras.layers import Lambda, Dense, MaxPooling2D, Input, Reshape, Conv2D
from keras.models import Model
from keras.callbacks import TensorBoard

# N = 5000 # individuals
# M = 100000 # SNPs
# K = 25 # latent dimension
#
#

def create_network(N, M, K):
	z = Input(shape=(N, K), name='z')
	w = Input(shape=(M, K), name='w')
	x = Input(shape=(N, M), name='x')
	y = Input(shape=(N, 1), name='y')
	e = Input(shape=(N, 1), name='epsilon') #noise

	# SNPs
	z_ = Reshape((N,1,K))(z)
	# Replace (-1,...) with (1,...) in backend.tile(-1,...)
	# Re: https://stackoverflow.com/questions/49743142/keras-lambda-invalidargumenterror-expected-multiples0-0-but-got-1
	#
	z_ = Lambda(lambda x: backend.tile(x, (1,1,M,1)))(z_) # N x M x K

	w_ = Reshape((1,M,K))(w)
	w_ = Lambda(lambda x: backend.tile(x, (1,N,1,1)))(w_) # N x M x K
	interim = Lambda(lambda x: backend.concatenate([x[0], x[1]], axis=3))([z_, w_])
	interim = Dense(512, activation='relu')(interim)
	print(interim.shape)
	interim = Conv2D(512, (10,10), padding='same', activation='relu', input_shape=(N, M, 512))(interim)
	interim = Dense(256, activation='relu')(interim)
	print(interim.shape)
	interim = Conv2D(256, (8,8), padding='same', activation='relu', input_shape=(N, M, 256))(interim)
	print(interim.shape)
	#interim = MaxPooling2D(pool_size=(2,2))(interim)
	interim = Dense(1)(interim)
	logits = Reshape((N,M))(interim)
	# print(interim.shape)

	snps_network = Model(inputs=[z, w], outputs=logits)
	snps_network.summary()
	plot_model(snps_network, to_file='snps_network.png', show_shapes=True)

	snps_network.compile(optimizer='adam', loss='mse')
	return snps_network

from snps_traits import load_data

# x = Alleles, y = traits, l = logits, z = confounders, w = priors
(trainx, trainy, trainl, trainz, trainw, testx, testy, testl, testz, testw) = load_data()

batch, N, K = trainz.shape
batch, M, _ = trainw.shape
snps_network = create_network(N, M, K)

epochs=80
batch_size=10
snps_network.fit(x=[trainz, trainw], y=trainl, epochs=epochs,
				 validation_data=([testz, testw], testl),
				 callbacks=[TensorBoard(log_dir='./tfb_logs/')])
snps_network.save_weights('icm_snps_traits_nw.h5')

import numpy as np
def comparison_svd(trainz, trainw, trainl):
	U, s, V = np.linalg.svd(trainl, full_matrices=False)

# Traits
# interim = Concatenate(axis=2)([z, x, e])
# interim = Dense(32, activation='relu')(interim)
# interim = Dense(256, activation='relu')(interim)
# interim = Concatenate(axis=2)([z, interim])
# outputs = Dense(1)(interim)
#
# traits_network = Model([z, x, e], outputs, name='traits_network')
# traits_network.summary()
# plot_model(traits_network, to_file='traits_network.png', show_shapes=True)
# reconstr_loss = mse(y,outputs)
#
# traits_network.compile(optimizer='adam', loss=reconstr_loss)

#
#  Tensorflow + edward code (wouldn't run because of incompatibilities between Tensorflow and
# Edward libraries
#
# def snp_neural_network(z, w):
# 	z_tile = tf.tile(tf.reshape(z, [N, 1, K]), [1, M, 1])
# 	w_tile = tf.tile(tf.reshape(w, [1, M, K]), [N, 1, 1])
# 	h = tf.concat([z_tile, w_tile], 2)
# 	h = tf.layers.dense(h, 512, activation=tf.nn.relu)
# 	h = tf.layers.dense(h, 512, activation=tf.nn.relu)
# 	h = tf.layers.dense(h, 1, activation=None)
# 	return tf.reshape(h, [N, M])

# def trait_neural_network(z, x):
# 	eps = np.random.normal(loc=0., scale=1., size=(N, 1))
# 	h = tf.concat([z, x, eps], 1)
# 	h = tf.layers.dense(h, 32, activation=tf.nn.relu)
# 	h = tf.layers.dense(h, 256, activation=tf.nn.relu)
# 	h = tf.concat([z, h], 1)
# 	h = tf.layers.dense(h, 1, activation=None)
# 	return tf.reshape(h, [N])
#
# z = np.random.normal(loc=0., scale=1., size=(N, K))
# w = np.random.normal(loc=0., scale=1., size=(M, K))
# logits = snp_neural_network(z, w)
# print(type(logits))
# x = binom(2, 1./(1. + tf.exp(-logits)))
#
# y = trait_neural_network(z, x)
