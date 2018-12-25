import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Turn off interactive plotting
#plt.ioff()

# nx = np.linspace(-np.pi,np.pi,20)
# X = np.cos(nx)
# #fig = plt.figure(figsize=(8,16))
# plt.plot(nx, X)
# plt.scatter(nx, X)
# #plt.savefig()
#
# M = np.outer(X, X)
# print(M)
# sns.heatmap(M)
# plt.show()

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv1D, MaxPool1D, UpSampling1D

nsamples = 5000
x_train = np.linspace(-2*np.pi, 2*np.pi, nsamples)
y_train = np.cos(x_train)
x_test = np.linspace(-np.pi, np.pi, 10)
y_test = np.cos(x_test)

np.random.seed(40997)
#Layers
my_layers = Sequential()
my_layers.add(Dense(32, input_shape=(1,), activation='relu'))
my_layers.add(Dense(1, activation='tanh'))
my_layers.summary()

#Model
input_data = Input(shape=(1,))
my_model = Model(input_data, my_layers(input_data))
my_model.summary()
my_model.compile(optimizer='adam', loss='mean_squared_error')

my_model.fit(x_train, y_train, epochs=10, batch_size=10, shuffle=True,validation_data=(x_test, y_test))
my_model.save('my_freq_covar.h5')
y_ = my_model.predict(x_test)

#Out of training support
x_out = np.linspace(np.pi,4*np.pi, 10)
y_out = my_model.predict(x_out)

plt.scatter(x_train, y_train, color='g',alpha=.1, label='Training Data')
plt.scatter(x_test, y_test, color='b',label='Validation Data')
plt.scatter(x_test, y_, color='k', label='Validation Prediction')
plt.scatter(x_out, y_out, color='r',label='Test Out of Support')
plt.tight_layout()
plt.legend()
plt.show()
