from keras.layers import Input, Dense
from keras.models import Model, Sequential
import time
#
# To change the backend from Theano to Tensorflow
# or the other way around. Edit ~/.keras/keras.json
#

start = time.time()
#
# by F. Chollet
# https://blog.keras.io/building-autoencoders-in-keras.html
#
encoding_dim = 32

#placeholder for input
input_img = Input(shape=(784,))

encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)
#
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# pixel level cross entropy loss
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Dataset
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
				batch_size=256,
				epochs=10,
				shuffle=True,
				validation_data=(x_test, x_test),
				verbose=False
				)

# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = autoencoder.predict(x_test)

print("Run completed in {:6.3f} secs".format(time.time() - start))

import matplotlib.pyplot as plt
n=10
plt.figure(figsize=(20,4))
for i in range(n):
	#original
	ax = plt.subplot(2, n, i+1)
	plt.imshow(x_test[i].reshape(28,28))
	plt.gray()
	# ax.get_xaxis().set_visible(False)
	# ax.get_yaxis().set_visible(False)

	#reconstruction
	ax = plt.subplot(2, n, i+1+n)
	plt.imshow(decoded_imgs[i].reshape(28,28))
	plt.gray()
	# ax.get_xaxis().set_visible(False)
	# ax.get_yaxis().set_visible(False)

plt.show()

#
# encoder_layers = Sequential()
# encoder_layers.add(Dense(32, activation='relu', input_dim=784))
# encoder_layers.add(Dense(1, activation='sigmoid'))
#
# decoder_layers = Sequential()
