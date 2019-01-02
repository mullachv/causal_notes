import numpy as np
from scipy.stats import binom
from scipy.stats import ortho_group
from scipy.special import expit

# Generate Synthetic SNPs and Traits
def gen_snps_traits(N, M, S):
	"""
	:param N: Number of observations (individuals), ranges in 100s to 10K
	:param M: Number of SNPs, ranges in 100K to 1M
	:param S: Sparsity factor (0,1]
	:return: tuple of confounder (Z), Prior (W), SNPs, Traits,
	"""

	def linear_combine(z, w):
		pi = np.dot(z, w.T)

		# Turn into a probability (sigmoid)
		return expit(pi)

	def gen_zwpi():
		#z = ortho_group.rvs(dim=N)
		k = int(M*S)
		z = np.random.randn(N, k)
		w = np.random.randn(M, k)
		pi = linear_combine(z, w)
		return z, w, pi

	def gen_xnm(pi_nm):
		return binom.rvs(2, pi_nm)

	def gen_traits(pi_nm, S='0.3'):
		q = int(pi_nm.shape[1] * (1.- S))
		zero_indexes = np.random.choice(pi_nm.shape[1], q)
		factor = np.random.randn(pi_nm.shape[1])
		for i in zero_indexes:
			factor[i] = 0.
		return np.dot(pi_nm, factor)

	z, w, probs = gen_zwpi()
	snps = gen_xnm(probs)
	return z, w, snps, gen_traits(probs, S), probs

import pandas as pd
def convert_snps_traits2df(snps, traits):
	individual_names = [('IND_' + str(x + 1)) for x in range(snps.shape[0])]
	df = pd.DataFrame(index=individual_names)
	for i in range(snps.shape[1]):
		ds = pd.Series(snps[:, i], index=individual_names, name=('SNP_' + str(i + 1)))
		df = df.join(ds)
	df = df.join(pd.Series(traits, index=individual_names, name='Traits'))
	return df

def array2df(data, row_pre='IND_', col_pre='Z_'):
	individual_names = [(row_pre + str(x + 1)) for x in range(data.shape[0])]
	df = pd.DataFrame(index=individual_names)
	for i in range(data.shape[1]):
		ds = pd.Series(data[:, i], index=individual_names, name=(col_pre + str(i + 1)))
		df = df.join(ds)
	return df

data_type_params = {
	'small': {
		'num_snips': 10**3,
		'num_individuals': 10,
		'inverse_sparsity': 40,
	},
	'medium': {
		'num_snips': 10**4,
		'num_individuals': 100,
		'inverse_sparsity': 60,
	},
	'large': {
		'num_snips': 10**5,
		'num_individuals': 5000,
		'inverse_sparsity': 300,
	},
}
data_folder = 'SNPS_TRAITS_data'
from pathlib import Path
def write_data(z, w, snips, traits, probs, j, hdf_file):
	df = convert_snps_traits2df(snips, traits)
	key_value = ('data' + '_' + str(j))
	df.to_hdf(hdf_file, key=key_value, mode='a')

	# logits
	logits = np.log(probs / (1. - probs))
	df = convert_snps_traits2df(logits, traits)
	key_value = ('/logits' + '_' + str(j))
	df.to_hdf(hdf_file, key=key_value, mode='a')

	# z
	df = array2df(z, row_pre='IND_', col_pre='Z_')
	key_value = ('/z' + '_' + str(j))
	df.to_hdf(hdf_file, key=key_value, mode='a')

	# w
	df = array2df(w, row_pre='AL_', col_pre='IND_')
	key_value = ('/w' + '_' + str(j))
	df.to_hdf(hdf_file, key=key_value, mode='a')

NUM_SETS = 3
def create_simple_data(data_type='small'):
	# create data, if not exist
	hdf_name = data_type + '.h5'
	hdf_file = Path(os.path.join(data_folder, hdf_name))
	if not (hdf_file.is_file()):  # .h5 file does not exist
		j, snip_size, individuals, inv_sparse_factor \
			= 0, \
				data_type_params[data_type]['num_snips'], \
				data_type_params[data_type]['num_individuals'], \
				data_type_params[data_type]['inverse_sparsity']

		if individuals == 0:
			return

		while j < NUM_SETS:
			# sparsity = 1./inv_sparse_factor
			z, w, snips, traits, probs = gen_snps_traits( individuals, snip_size, 1. / inv_sparse_factor)
			# print('w.shape: {}'.format(w.shape))
			write_data(z, w, snips, traits, probs, j, hdf_file)
			j += 1

	return

def get_entry_key(hdf_keys, data_indicator, train_test='train'):
	c = 0 if train_test == 'train' else -1
	return list(filter(lambda x: x.startswith('/' + data_indicator), hdf_keys))[c]

def read_data(data_type='small'):
	hdf_name = data_type + '.h5'
	hdf_fname = os.path.join(data_folder, hdf_name)
	hdf_file_path = Path(hdf_fname)
	if not (hdf_file_path.is_file()):  # .h5 file does not exist
		raise Exception("H5 file does not exist: " + hdf_file_path)
	hf = pd.HDFStore(hdf_fname, mode='r')
	data_keys = hf.keys()
	data_keys.sort()

	acc = []

	for text in ['train', 'test']:
		read_key = get_entry_key(data_keys, 'data', text)
		df = pd.read_hdf(hdf_file_path, key=read_key, mode='r')
		_, ncols = df.shape

		read_key = get_entry_key(data_keys, 'logits', text)
		dl = pd.read_hdf(hdf_file_path, key=read_key, mode='r')

		read_key = get_entry_key(data_keys, 'z', text)
		dz = pd.read_hdf(hdf_file_path, key=read_key, mode='r')

		read_key = get_entry_key(data_keys, 'w', text)
		dw = pd.read_hdf(hdf_file_path, key=read_key, mode='r')

		acc += df.ix[:, :ncols - 1],
		acc += df.ix[:, ncols - 1],
		acc += dl.ix[:, :ncols - 1],
		acc += dz.ix[:],
		acc += dw.ix[:],

	# train, test
	# (x_train, y_train, l_train, z_train, w_train, x_test, y_test, l_test, z_test, w_test)
	return acc


import numpy as np
def make_batch(data_frame):
	sh = [-1] + [x for x in data_frame.shape]
	return np.reshape(data_frame.values, sh)

import os
def load_data(data_type='small'):
	data_folder = "SNPS_TRAITS_data"
	# Make dir, if not exist
	os.makedirs(data_folder, exist_ok=True)

	# data_types = ['small', 'medium', 'large']
	create_simple_data(data_type)

	# Load data into a pd
	x, y, l, z, w, sx, sy, sl, sz, sw = read_data(data_type)
	return make_batch(x), \
			make_batch(y), \
			make_batch(l), \
			make_batch(z), \
			make_batch(w), \
			make_batch(sx), \
			make_batch(sy), \
			make_batch(sl), \
			make_batch(sz),	\
			make_batch(sw)

import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-D', "--data_type", help="small, medium or large", choices=['small', 'medium', 'large'],
						default='small', action='store')
	# parser.add_argument('-A', "--all", help="Generate all prebuilt synthetic datasets", default=True, action='store_true')
	# parser.add_argument('-M', "--snip_size", help="Number of genomic SNPs", default=30000)
	# parser.add_argument('-N', '--observations', help="Number of genomic observations", default=100)
	# parser.add_argument('-S', '--sparsity', help="Sparsity", default=0.3)
	# parser.add_argument('-R', '--relationship', help="Relationship b/w SNPs and traits", default='linear')

	args = parser.parse_args()
	(trainx, trainy, trainl, trainz, trainw, testx, testy, testl, testz, testw) = load_data(args.data_type)
	print(trainy.shape, testy.shape)
	print(trainl.shape, testl.shape)
	print(trainz.shape, trainw.shape)
