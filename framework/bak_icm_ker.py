from keras import backend
from keras.datasets import mnist
from keras.utils import plot_model
from keras.layers import Lambda, Dense, Concatenate, Input, Reshape, RepeatVector
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.callbacks import TensorBoard

N = 5000 # individuals
M = 100000 # SNPs
K = 25 # latent dimension

z = Input(shape=(N, K), name='z')
w = Input(shape=(M, K), name='w')
x = Input(shape=(N, M), name='x')
y = Input(shape=(N, 1), name='y')
e = Input(shape=(N, 1), name='epsilon') #noise

# Preprocess
def preprocess_zw(args):
	z, w = args
	input1 = backend.reshape(z, (N, 1, K)) # N x 1 x K
	input1  = backend.tile(input1, (1, M, 1)) # N x M x K

	input2 = backend.reshape(w, (1, M, K))
	input2 = backend.tile(input2, (N, 1, 1)) # N x M x K

	real_input = backend.concatenate([input1, input2], axis=2)
	#print(real_input.shape)
	return real_input

def squeeze_reshape(x):
	x = backend.squeeze(x, axis=2)
	x = backend.reshape(x, [N, M])
	return x

# SNPs
interim = Lambda(preprocess_zw)([z, w])
print(interim.shape)
interim = Dense(512, activation='relu')(interim)
interim = Dense(512, activation='relu')(interim)
interim = Dense(1)(interim)
print(interim.shape)
logits = Lambda(squeeze_reshape)(interim)

snps_network = Model(inputs=[z, w], outputs=logits)
snps_network.summary()
plot_model(snps_network, to_file='snps_network.png', show_shapes=True)

# x_pred = backend.random_binomial(logits.shape, backend.sigmoid(logits))
#cat_loss = backend.sparse_categorical_crossentropy(x, logits, from_logits=True)
cat_loss = backend.binary_crossentropy(x, logits, from_logits=True)
print(cat_loss.shape)
print(type(cat_loss))

from snps_traits import load_data
(trainx, trainy, trainl, testx, testy, testl) = load_data()

snps_network.compile(optimizer='adadelta', loss=backend.mean(cat_loss))


epochs=3
batch_size=10
snps_network.fit(trainx, trainy, epochs=epochs, batch_size=batch_size,
				 validation_data=(testx, testy),
				 callbacks=[TensorBoard(log_dir='./tfb_logs/')])

# Traits
interim = Concatenate(axis=2)([z, x, e])
interim = Dense(32, activation='relu')(interim)
interim = Dense(256, activation='relu')(interim)
interim = Concatenate(axis=2)([z, interim])
outputs = Dense(1)(interim)

traits_network = Model([z, x, e], outputs, name='traits_network')
traits_network.summary()
plot_model(traits_network, to_file='traits_network.png', show_shapes=True)
reconstr_loss = mse(y,outputs)

traits_network.compile(optimizer='adam', loss=reconstr_loss)


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
