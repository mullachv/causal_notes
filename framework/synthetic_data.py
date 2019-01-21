import numpy as np
np.random.seed(4099777)
from scipy.stats import binom
import pandas as pd

#
# https://stackoverflow.com/questions/12297016/how-to-find-most-frequent-values-in-numpy-ndarray?rq=1
#
def most_frequent(x):
    u , indices = np.unique(x, return_inverse=True)
    axis = 0
    return u[np.argmax(
                    np.apply_along_axis(np.bincount,
                            axis,
                            indices.reshape(x.shape),
                            None,
                            np.max(indices)+1
                        ),
                axis=axis)]

RINSE_N_REPEAT = 1000
def create_individual_snp(M):
	probs = np.random.random([M])
	x = []
	for i in range(RINSE_N_REPEAT):
		x += binom.rvs(2, probs).tolist(),
	x = np.asarray(x, int)
	return most_frequent(x).tolist(), probs

def create_df_from2D_arr(arr, column_pre='ATTR_', arr_dtype='str'):
	assert len(arr.shape) == 2
	individual_names = [('IND_' + str(x + 1)) for x in range(arr.shape[0])]
	df = pd.DataFrame(index=individual_names)
	for i in range(arr.shape[1]):
		colname = column_pre + str(i + 1)
		df[colname] = arr[:,i]
		df[colname].astype(arr_dtype)
	return df

def create_df_from1D_arr(arr, column_pre='ATTR', arr_dtype='str'):
	assert len(arr.shape) == 1
	individual_names = [('IND_' + str(x + 1)) for x in range(arr.shape[0])]
	df = pd.DataFrame(index=individual_names)
	colname = column_pre
	df[colname] = arr[:]
	df[colname].astype(arr_dtype)
	return df

def create_df_from_arr(arr, column_pre='ATTR_', arr_dtype='str'):
	assert len(arr.shape) == 1 or 2
	if len(arr.shape) == 2:
		return create_df_from2D_arr(arr, column_pre, arr_dtype)
	return create_df_from1D_arr(arr, column_pre, arr_dtype)

def create_snps(N, M):
	probs = []
	x = []
	for _ in range(N):
		x1, prob = create_individual_snp(M)
		x += x1,
		probs += prob,

	return create_df_from_arr(np.asarray(x, int), 'SNPS_', 'int'), create_df_from_arr(np.asarray(probs), 'PROB_', 'float32')
	#return np.asarray(x, int), np.asarray(probs)

def create_traits(N):

	# Continuous
	# return create_df_from_arr(binom.rvs(1, np.random.randn(N)), 'TRAITS', 'float32')

	# Discrete
	return create_df_from_arr(binom.rvs(1, np.random.random(N)), 'TRAITS', 'int')
	#return np.random.randn(N)

def logits(pi_nm):
	return np.log(pi_nm/(1. - pi_nm))

def combine_arrs2df(arr1, arr2=None, arr1_column_pre='ARR1_', arr1_dtype='str', arr2_column_pre='ARR2_', arr2_dtype='str'):
	assert len(arr1.shape) == 2
	assert arr1.shape[0] == arr2.shape[0]
	individual_names = [('IND_' + str(x + 1)) for x in range(arr1.shape[0])]
	df = pd.DataFrame(index=individual_names)
	for i in range(arr1.shape[1]):
		colname = arr1_column_pre + str(i + 1)
		df[colname] = arr1[:,i]
		df[colname].astype(arr1_dtype)

	for i in range(arr2.shape[1]):
		colname = arr2_column_pre + str(i + 1)
		df[colname] = arr2[:, i]
		df[colname].astype(arr2_dtype)

	return df

def create_non_gwas_observed_data(N, D1, D2):
	def create_contns_data(N, D1):
		return np.random.randn(N, D1)

	NUM_DISCRETE_CLASSES = 3
	def create_discrete_data(N, D2):
		return binom.rvs(NUM_DISCRETE_CLASSES - 1, np.random.random([N, D2]))

	# def create_categorical_data(N, D3):
	# 	pass

	return combine_arrs2df(create_contns_data(N, D1),
						   create_discrete_data(N, D2),
						   arr1_column_pre='CONT_',
						   arr1_dtype='float32',
						   arr2_column_pre='DISC_',
						   arr2_dtype='int'
						   )
	# return np.concatenate((create_contns_data(N, D1), create_discrete_data(N, D2).astype(int)), axis=1)

import os
data_folder = 'SYNTHETIC_DATA'
data_filename=os.path.join(data_folder + '/simulation.h5')
def store_data(df, data_key, **kwargs):
	os.makedirs(data_folder, exist_ok=True)
	store = pd.HDFStore(data_filename, mode='a')
	store.put(data_key, df, format='fixed', data_columns=True)
	#
	# table format is required for batch generator (chunked reading)
	# however, writing fails in that mode
	# 
	#
	store.put(data_key, df, format='table', data_columns=True)
	store.close

def _int_load_data(data_key='/trn_data_1000x1000__x_nm'):
	with pd.HDFStore(data_filename) as store:
		data = store[data_key]
		if data.empty:
			raise Exception("Key Not found in HDFStore: " + data_key)
		#metadata = store.get_storer(data_key).attrs.metadata
	return data
	#return data, metadata

def make_batch(data_frame):
	sh = [-1] + [x for x in data_frame.shape]
	return np.reshape(data_frame.values, sh)

Kc = 40
#
# https://stackoverflow.com/questions/20111542/selecting-rows-from-an-hdfstore-given-list-of-indexes
#
def batch_generator(batch_size=128, data_key_pre=None, dt='/trn'):
	with pd.HDFStore(data_filename) as store:
		if dt not in ['/trn', '/val', '/tst']:
			raise Exception("Invalid data type: {}. Must be one of '/trn', '/val', or '/tst'".format(dt))
		if data_key_pre is None or data_key_pre == '':
			pre = [u for u in store.keys() if u.startswith(dt)][0]
			data_key_pre = pre[:pre.find('__')]

		while True:
			nrows = store.get_storer(data_key_pre + '__o_k').shape[0]
			indices = np.random.randint(nrows, size=batch_size)
			o = store.select(key=data_key_pre + '__o_k', where='index in indices', chunksize=batch_size)
			y = store.select(key=data_key_pre + '__y_n', where='index in indices', chunksize=batch_size)
			pi = store.select(key=data_key_pre + '__pi_nm', where='index in indices', chunksize=batch_size)
			x = store.select(key=data_key_pre + '__x_nm', where='index in indices', chunksize=batch_size)
			x, pi, y, o = make_batch(x), make_batch(pi), make_batch(y), make_batch(o)
			yield [x, o], y


def load_data(data_key_pre=None, dt='/trn'):
	'''
	:param data_key_pre: 'data_1000x1000__'
	:return: x, pi, y, o data as a tuple of each of one batch
	'''
	with pd.HDFStore(data_filename) as store:
		if dt not in ['/trn', '/val', '/tst']:
			raise Exception("Invalid data type: {}. Must be one of '/trn', '/val', or '/tst'".format(dt))
		if data_key_pre is None or data_key_pre == '':
			pre = [u for u in store.keys() if u.startswith(dt)][0]
			data_key_pre = pre[:pre.find('__')]
		o = store[data_key_pre + '__o_k']
		y = store[data_key_pre + '__y_n']
		pi = store[data_key_pre + '__pi_nm']
		x = store[data_key_pre + '__x_nm']

		if o.empty or y.empty or pi.empty or x.empty:
			raise Exception("Incomplete data found in the H5 file: {}. "
							"Delete and create new H5 file by running: {}".format(data_filename,'synthetic_data.py'))
		o = make_batch(o)
		y = make_batch(y)
		pi = make_batch(pi)
		x = make_batch(x)

	return x, pi, y, o

def load_training_data(data_key_pre=None):
	return load_data(data_key_pre, '/trn')

def load_validation_data(data_key_pre=None):
	return load_data(data_key_pre, '/val')

def load_test_data(data_key_pre=None):
	return load_data(data_key_pre, '/tst')

def update_data_folder(root):
	global data_folder, data_filename
	data_folder = os.path.join(args.data_root + '/' + data_folder)
	data_filename = os.path.join(data_folder + '/simulation.h5')

import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-D', "--data_root", help="Root data folder", default='./', type=str)
	parser.add_argument('-L', "--m_low", help="Low range of SNPs (exp scale)", default=3, type=float)
	parser.add_argument('-H', "--m_hi", help="High range of SNPs (exp scale)", default=4, type=float)
	parser.add_argument('-N', "--num_m", help="SNP count set size", default=2, type=int)
	parser.add_argument('-V', "--n_low", help="Low range of individuals", default=100, type=int)
	parser.add_argument('-W', "--n_hi", help="High range of individuals", default=500, type=int)
	parser.add_argument('-S', "--incr_n", help="Individual count increment", default=400, type=int)
	parser.add_argument('-C', "--num_cntns", help="Number of observed data of type continuous", default=40, type=int)
	parser.add_argument('-T', "--num_discr", help="Number of observed data of type discrete", default=10, type=int)

	args = parser.parse_args()
	update_data_folder(args.data_root)

	if not (os.path.exists(data_filename)):
		for M in np.logspace(args.m_low, args.m_hi, num=args.num_m):
			M = int(M)
			for N in range(args.n_low, args.n_hi, args.incr_n):
				for ds in ['trn', 'val', 'tst']:
					data_key_pre = ds + '_data_' + str(N) + 'x' + str(M) + '__'
					x_nm, pi_nm = create_snps(N, M)
					store_data(x_nm, data_key_pre + 'x_nm', shape=x_nm.shape)
					store_data(pi_nm, data_key_pre + 'pi_nm', shape=pi_nm.shape)
					y_n = create_traits(N)
					store_data(y_n, data_key_pre + 'y_n', shape=y_n.shape)
					# K = 50
					o_k = create_non_gwas_observed_data(N, args.num_cntns, args.num_discr)
					store_data(o_k, data_key_pre + 'o_k', shape=o_k.shape)
			print("Data create and stored for snip size = {}".format(M))

	# internal testing
	data = _int_load_data('/trn_data_100x1000__x_nm')

	x, pi, y, o = load_training_data()
	x, pi, y, o = load_validation_data()
	x, pi, y, o = load_test_data()

	# x, pi, y, o = batch_generator(10)
	# print(x)

#  x_nm:
# 	[[1 1 2 2 0]
#  		[0 1 0 2 2]
#  		[1 2 0 0 2]]
#
#  pi_nm:
# [[ 0.52602119  0.48006654  0.69422182  0.74390746  0.15667243]
#  [ 0.21494761  0.53392421  0.0152188   0.8729723   0.89140151]
#  [ 0.35717252  0.72334727  0.11496124  0.05161208  0.93346941]]
#
#  Traits:
# [ 0.18728446  1.05313792  0.78911622]
#
#
