from keras import backend
from keras.utils import plot_model
from keras.layers import Lambda, Dense, Input, Reshape, Embedding
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.losses import mse
import keras.metrics as metrics

import argparse
import numpy as np

from synthetic_data import load_training_data

N = 100 # individuals
M = 1000 # SNPs
# K = Kc + Kd -- Observed data dimension
Kc = 40
Kd = 10
H = 10 # Latent dimension

def reshape_rebatch(x, o, y):
	r = x.shape[2] # M
	assert len(x.shape) == 3 # Batch=1, N, M
	x = x.swapaxes(0,2) # Batch=M, N, 1

	assert len(o.shape) == 3 # 1, N, K=50
	o = np.repeat(o, r, axis=0) # M, N, K

	assert len(y.shape) == 3 # 1, N, 1
	y = np.tile(y, r)
	y = y.swapaxes(0, 2) # Batch=M, N, 1

	# We have M batches of data
	return x, o, y

#
# Reparameterization trick
# z = z_mean + var * epsilon
# where epsilon ~ Normal(0,1)
#
def sampling(args):
	z_mean, z_log_var = args
	batch = backend.shape(z_mean)[0]
	dim1 = backend.int_shape(z_mean)[1]
	dim2 = backend.int_shape(z_mean)[2]
	epsilon = backend.random_normal(shape=(batch, dim1, dim2))
	return z_mean + backend.exp(0.5 * z_log_var)*epsilon

# Read Train and reshape
# x = Alleles, y = traits, pi = probs, o = observations
#
train_x, train_pi, train_y, train_o = load_training_data()
train_x, train_o, train_y = reshape_rebatch(train_x, train_o, train_y)


# Parameters
num_discrete_classes = 3
# x = Input(shape=(N, num_discrete_classes), name='x_as_categorical')
x = Input(shape=(N, 1,), name='x_as_categorical')
oc = Input(shape=(N, Kc,), name='o_continuous')
od = Input(shape=(N, num_discrete_classes * Kd,), name='o_discrete') # one for each discrete o, Kd
y = Input(shape=(N, 1,), name='y') # traits, continuous

##### Encoder ######
interim = Dense(512, activation='relu')(x)
interim = Dense(512, activation='relu')(interim)
interim = Dense(256, activation='relu')(interim)
z = Dense(50)(interim) # ?, N, 50

# interim = Lambda(lambda x: backend.concatenate([x[0], x[1]], axis=2))([oc, od])
interim = Dense(512, activation='relu')(oc)
interim = Dense(512, activation='relu')(interim)
interim = Dense(256, activation='relu')(interim)
z_ = Dense(50)(interim) # ?, N, 50

#
latent_dim = 10
interim = Lambda(lambda x: backend.concatenate([x[0], x[1]], axis=2))([z, z_]) # ?, N, 100
interim = Dense(512, activation='relu')(interim)
interim = Dense(256, activation='relu')(interim)
u_mean = Dense(latent_dim, name='u_mean')(interim) # ?, N, 10
u_log_var = Dense(latent_dim, name='u_log_var')(interim) # ?, N, 10

# Sample
u = Lambda(sampling, output_shape=(N,latent_dim,))([u_mean, u_log_var])

# Encoder Model
encoder = Model([x, oc], [u_mean, u_log_var, u], name='encoder')
encoder.summary()
plot_model(encoder, to_file='cmed_encoder.png', show_shapes=True)

####### Decoder #####
latent_inputs = Input(shape=(N, latent_dim,), name='u_sampling')
interim = Dense(512,activation='relu')(latent_inputs)
interim = Dense(256,activation='relu')(interim)
outputs = Dense(1)(interim)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='cmed_decoder.png', show_shapes=True)

inputs = [x, oc]
outputs = decoder(encoder(inputs)[2])
cmed_vae = Model(inputs, outputs, name='vae_cmed')
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-w', '--weights', help="Saved model weights file")
	args = parser.parse_args()

	#
	# https://github.com/keras-team/keras/issues/10137
	# Compute VAE loss (important to have loss in a function
	#
	def my_vae_loss(y_true, y_pred):
		xent_loss = N * \
					1 * \
					metrics.binary_crossentropy(
						backend.flatten(y_true),
						backend.flatten(y_pred))
		kl_loss = - 0.5 * backend.sum(1 + u_log_var - backend.square(u_mean) - backend.exp(u_log_var), axis=-1)
		vae_loss = backend.mean(xent_loss + kl_loss)

		return vae_loss

	cmed_vae.compile(optimizer='adam', loss=my_vae_loss)
	cmed_vae.summary()

	plot_model(cmed_vae, to_file='cmed_mlp.png', show_shapes=True)

	if args.weights:
		cmed_vae.load_weights(args.weights)
	else:
		# train
		cmed_vae.fit(x=[train_x, train_o[:,:,:Kc]], y=train_y, epochs=3, batch_size=32) #,
#					 validation_data=(train_x, None))
		cmed_vae.save_weights('cmed_syn_model.h5')
