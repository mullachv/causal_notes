import numpy as np
import time
from keras.layers import Dense
from keras.models import Model, Input

np.random.seed(40997)

from snps_traits import gen_snps_traits
M, N, K = 100, 5, 2
snps, traits = gen_snps_traits(M, N, K)

start_time = time.time()

input_ = Input(shape=(M,))
encoded = Dense(1000, activation='relu')(input_)
decoded = Dense(input_, activation='sigmoid')(encoded)

