from keras import backend
from keras.utils import plot_model
from keras.layers import Lambda, Dense, Input, Dropout, Embedding, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.losses import mse
import keras.metrics as metrics

import argparse
import numpy as np

from synthetic_data import load_training_data, load_validation_data, load_test_data

N = 100 # individuals
M = 10000 # SNPs
# K = Kc + Kd -- Observed data dimension
Kc = 40
Kd = 10
H = 10 # Latent dimension
NUM_DISC_CLASSES = 3
dropout_rate = 0.3

def reshape_rebatch(x, o, y):
	assert len(x.shape) == 3
	x = np.squeeze(x, axis=0) # Batch, M

	assert len(o.shape) == 3
	o = np.squeeze(o, axis=0) # Batch, K

	assert len(y.shape) == 3
	y = np.squeeze(y, axis=0) # Batch, 1

	# We have N batches of data
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
	epsilon = backend.random_normal(shape=(batch, dim1))
	return z_mean + backend.exp(0.5 * z_log_var)*epsilon

# Read Train and reshape
# x = Alleles, y = traits, pi = probs, o = observations
#
tx, _, ty, to = load_training_data()
vx, _, vy, vo = load_validation_data()
tx, to, ty = reshape_rebatch(tx, to, ty)
vx, vo, vy = reshape_rebatch(vx, vo, vy)


# Parameters
x = Input(shape=(M,), name='x_as_categorical')
oc = Input(shape=(Kc,), name='o_continuous')
od = Input(shape=(Kd,), name='o_discrete') # one for each discrete o, Kd
y = Input(shape=(1,), name='y') # traits, continuous

### Encoder ###
# input x
# Embedding [0,1,2] - 3 classes of input. Output 2D vector
interim = Embedding(NUM_DISC_CLASSES, 2, input_length=M)(x) #output of batch, M, 2
interim = Dense(512, activation='relu')(interim)
interim = Dropout(dropout_rate)(interim)
interim = Dense(256, activation='relu')(interim)
interim = Flatten()(interim)
z1 = Dense(50)(interim) # ?, 50

# input oc
interim = Dense(512, activation='relu')(oc)
interim = Dropout(dropout_rate)(interim)
interim = Dense(256, activation='relu')(interim)
z2 = Dense(50)(interim) # ?, 50

# input od
# Embedding 3 classes of input. Output 2D vector
interim = Embedding(NUM_DISC_CLASSES, 2, input_length=Kd)(od) #output of batch, Kd, 2
interim = Dense(512, activation='relu')(interim)
interim = Dropout(dropout_rate)(interim)
interim = Dense(256, activation='relu')(interim)
interim = Flatten()(interim)
z3 = Dense(50)(interim) # ?, 50

# input [z1 z2 z3]
latent_dim = 10
interim = Lambda(lambda x: backend.concatenate([x[0], x[1], x[2]], axis=1))([z1, z2, z3]) # ?, 150
interim = Dense(512, activation='relu')(interim)
interim = Dropout(dropout_rate)(interim)
interim = Dense(256, activation='relu')(interim)
u_mean = Dense(latent_dim, name='u_mean')(interim) # ?, 10
u_log_var = Dense(latent_dim, name='u_log_var')(interim) # ?, 10

# Sample
u = Lambda(sampling, output_shape=(latent_dim,))([u_mean, u_log_var])
print(u.shape)

# Encoder Model
encoder = Model([x, oc, od], [u_mean, u_log_var, u], name='encoder')
encoder.summary()
plot_model(encoder, to_file='cmed_encoder.png', show_shapes=True)

### Decoder ###
latent_inputs = Input(shape=(latent_dim,), name='u_sampling')
interim = Dense(512,activation='relu')(latent_inputs)
interim = Dropout(dropout_rate)(interim)
interim = Dense(256,activation='relu')(interim)
outputs = Dense(1)(interim)
print(outputs.shape)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='cmed_decoder.png', show_shapes=True)

inputs = [x, oc, od]
outputs = decoder(encoder(inputs)[2]) # decoder input is u from [u_mean, u_log_var, u], so index 2
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

	cmed_vae.compile(optimizer='adam', loss=my_vae_loss, metrics=['accuracy'])
	cmed_vae.summary()

	plot_model(cmed_vae, to_file='cmed_mlp.png', show_shapes=True)

	if args.weights:
		cmed_vae.load_weights(args.weights)
	else:
		# train
		cmed_vae.fit(x=[tx, to[:, :Kc], to[:, Kc:]],
					 y=ty,
					 epochs=3,
					 batch_size=10,
					 validation_data=([vx, vo[:, :Kc], vo[:, Kc:]], vy)
					 )
		cmed_vae.save_weights('cmed_syn_model.h5')
