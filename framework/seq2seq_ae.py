from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector

timesteps=20
input_dim=10
latent_dim=20
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

# Find a dataset now, fetch x, y

# Compile

# fit (train), validate

#
#  predict and test/visualize
#
