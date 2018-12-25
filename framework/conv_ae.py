from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Input, MaxPooling2D, UpSampling2D
from keras import backend as K

input_img = Input(shape=(28, 28, 1)) #adapt this if using 'channels_first' image data format
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

#the representation is now (4,4,8) i.e. 128-dimensional vector
# Why?
# Because each Conv2D with padding does not alter the dimension of the
# image. Each MaxPooling2D(2,2) reduces the size by a factor of 2.
# So, we have: 28 / 2. = 14, 14/ 2. = 7, 7/ 2. = 4 (rounded)
# So = 4, 4, 8 (8 filters in the previous Conv2D)
#
x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Train
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

from keras.callbacks import TensorBoard
autoencoder.fit(x_train, x_train, batch_size=128,epochs=3,
				shuffle=True,validation_data=(x_test, x_test),
				callbacks=[TensorBoard(log_dir='./tfb_logs/')])

autoencoder.save('./conv_ae_model.h5')

# from keras.models import load_model
# autoencoder = load_model('./conv_ae_model.h5')
decoded_imgs = autoencoder.predict(x_test,batch_size=128)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
	#original
	ax = plt.subplot(2, n, i+1)
	plt.imshow(x_test[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# reconstruction
	ax = plt.subplot(2, n, i + 1+ n)
	plt.imshow(decoded_imgs[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('./conv_ae_reconstr.png')
plt.close()

plt.figure(figsize=(20,8))
for i in range(n):
	ax = plt.subplot(1, n, i+1)
	plt.imshow(encoded[i].reshape(4, 4*8).T)
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
