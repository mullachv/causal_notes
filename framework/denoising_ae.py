from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

input_img = Input(shape=(28, 28, 1)) #adapt this if using 'channels_first' image data format
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

denoising_autoencoder = Model(input_img, decoded)
denoising_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor*np.random.normal(loc=0., scale=1.,size=x_train.shape)
x_test_noisy = x_test + noise_factor*np.random.normal(loc=0., scale=1.,size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy,0.,1.)
x_test_noisy = np.clip(x_test_noisy,0.,1.)

from keras.callbacks import TensorBoard
denoising_autoencoder.fit(x_train_noisy, x_train, batch_size=128, epochs=3,
						  shuffle=True, validation_data=(x_test_noisy, x_test),
						  callbacks=[TensorBoard(log_dir='./tfb_logs/')])

denoising_autoencoder.save('./denoising_conv_ae_model.h5')
denoised_imgs = denoising_autoencoder.predict(x_test_noisy)

import matplotlib.pyplot as plt
n=10
for i in range(n):
	#noisy original
	ax = plt.subplot(2,n,i+1)
	plt.imshow(x_test_noisy[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# denoised
	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(denoised_imgs[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
