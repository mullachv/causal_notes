from keras.layers import Input, Dense, Concatenate
from keras.models import Model
import numpy as np

M = 1000
N = 5
Kc = 4

def fn(x1, x2):
	return np.sum(x1*2. + 3.*x2, axis=2)[..., np.newaxis]

x1 = np.random.randn(M, N, 1)
x2 = np.random.randn(M, N, Kc)
y1 = fn(x1, x2)

print(x1.shape)
print(x2.shape)
print(y1.shape)

val1 = np.random.rand(2, N, 1)
val2 = np.random.rand(2, N, Kc)
y2 = fn(val1, val2)

first = Input(shape=(N, 1,))
second = Input(shape=(N, Kc,))

interim1 = Dense(128, activation='relu')(first)
output = Dense(1)(interim1)

# interim2 = Dense(128, activation='relu')(second)
# interim2 = Dense(1)(interim2)
# s = Input(shape=(1))
# output = Concatenate([s, s])(interim1, interim2)

mp = Model(inputs=[first, second], outputs=output)
mp.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

mp.fit(x=[x1, x2], y=y1, batch_size=3,epochs=5, validation_data=([val1, val2], y2))

def distance(pred, actual):
	d = 0
	for i in range(pred.shape[0]):
		d += np.dot(pred[i][:,0], actual[i][:,0])
	print(d)
	return d * 1. / pred.shape[0]

pred = (mp.predict([val1, val2]))
print(distance(pred, y2))
