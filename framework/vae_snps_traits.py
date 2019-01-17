from keras.layers import Lambda, Dense, MaxPooling2D, UpSampling2D, Input
from keras.models import Model
from keras.losses import mse, binary_crossentropy
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.datasets import mnist
from keras.utils import plot_model
import argparse
import os

# Reparameterization trick
# z = z_mean + var * epsilon
# where epsilon ~ Normal(0,1)
#
def sampling(args):
	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.shape(z_mean)[1]
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var)*epsilon

def plot_results(models, data, batch_size=128, model_name='vae_mnist'):
	"""
	Plot labels and MNIST digits as function of 2-dim latent vector

	:param models: tuple of encode and decoder models
	:param data: tuple of test data and label
	:param batch_size: prediction batch size
	:param model_name: calling model name
	:return: None
	"""
	encoder, decoder = models
	x_test, y_test = data
	os.makedirs(model_name, exist_ok=True)

	filename = os.path.join(model_name, "vae_mean.jpg")
	# display a 2D plot of digit classes in the latent space
	z_mean, _, _ = encoder.predict(x_test,batch_size=batch_size)
	plt.figure(figsize=(12,10))
	plt.scatter(z_mean[:,0], z_mean[:,1], c=y_test)
	plt.colorbar()
	plt.xlabel("z[0]")
	plt.ylabel("z[1]")
	plt.savefig(filename)
	plt.show()

	filename = os.path.join(model_name, "digits_over_latent.jpg")
	# display a 30x30 2-D manifold of digits
	n = 30
	digit_size = 28
	figure = np.zeros((digit_size*n, digit_size*n))
	# linearly spaced coordinates corresponding to the 2D plot
	# of digits classes in latent space
	grid_x = np.linspace(-4,4,n)

	#reversed of above
	grid_y = np.linspace(-4,4,n)[::-1]

	for i, yi in enumerate(grid_y):
		for j, xi in enumerate(grid_x):
			z_sample = np.array([[xi, yi]])
			x_decoded = decoder.predict(z_sample)
			digit = x_decoded[0].reshape(digit_size, digit_size)
			figure[i*digit_size : (i+1)*digit_size, j*digit_size : (j+1)*digit_size] = digit

	plt.figure(figsize=(10,10))
	start_range = digit_size // 2
	end_range = n * digit_size + start_range + 1
	pixel_range = np.arange(start_range, end_range, digit_size)
	sample_range_x = np.round(grid_x, 1)
	sample_range_y = np.round(grid_y, 1)
	plt.xticks(pixel_range, sample_range_x)
	plt.yticks(pixel_range, sample_range_y)
	plt.xlabel("z[0]")
	plt.ylabel("z[1]")
	plt.imshow(figure, cmap="Greys_r")
	plt.savefig(filename)
	plt.show()

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 4

# VAE model = encoder + decoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# sample
# output shape is not required with tensorflow backend
# so we can write Lambda(sampling)([z_mean, z_log,sigma])
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim,activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
print(outputs.shape)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
print(outputs.shape)
vae = Model(inputs, outputs,name='vae_mlp')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	help_ = 'Load h5 model trained weights'
	parser.add_argument('-w', '--weights',help=help_)
	help_= 'Use mse loss instead of binary cross entropy'
	parser.add_argument('-m', '--mse', help=help_, action='store_true')

	args = parser.parse_args()
	models = (encoder, decoder)
	data = (x_test, y_test)

	# VAE loss =
	# mse_loss + kl_loss
	# or
	# xentropy_loss + kl_loss
	#
	if args.mse:
		reconstruction_loss = mse(inputs, outputs)
	else:
		reconstruction_loss = binary_crossentropy(inputs, outputs)

	reconstruction_loss *= original_dim

	# KL loss
	# KL(P||Q), when P~N(mu, var), and Q ~ N(0,1):
	# -1/2(var + mu^2 -1 - log var)
	# re: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
	#
	kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
	kl_loss = K.sum(kl_loss, axis=-1)
	kl_loss *= -0.5

	vae_loss = K.mean(reconstruction_loss + kl_loss)
	vae.add_loss(vae_loss)
	vae.compile(optimizer='adam')
	vae.summary()

	print('reconstr_loss shape: {}'.format(reconstruction_loss.shape))
	print('kl_loss shape: {}'.format(kl_loss.shape))
	print('vae_loss shape: {}'.format(vae_loss.shape))


	plot_model(vae, to_file='vae_mlp.png',show_shapes=True)

	if args.weights:
		vae.load_weights(args.weights)
	else:
		# train
		vae.fit(x_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test, None))
		vae.save_weights('vae_mlp_mnist.h5')

	plot_results(models, data, batch_size=batch_size,model_name='vae_mlp')


