import numpy as np
from scipy.stats import binom

# Generate Synthetic SNPs and Traits
def gen_snps_traits(M, N, S):
	"""

	:param M: Number of SNPs, ranges in 100K to 1M
	:param N: Number of observations (individuals), ranges in 100s to 10K
	:param S: Sparsity factor (0,1]
	:return: tuple of SNPs and Traits
	"""
	def gen_pi_nm():
		return np.random.rand(N, M)

	def gen_xnm(pi_nm):
		return binom.rvs(2, pi_nm)

	pi_nm = gen_pi_nm()
	snps = gen_xnm(pi_nm)

	vars_coeff_ind, vars_coeff = 0, None

	# What fraction of M's contribute to y_n
	# Probability is compressed
	prob_compression_factor = S

	def coeffs():
		nonlocal vars_coeff_ind, vars_coeff
		if vars_coeff_ind == 0:
			vars_coeff_ind = 1
			a = binom.rvs(1, [prob_compression_factor]*pi_nm.shape[1])
			b = np.random.randn(pi_nm.shape[1])
			vars_coeff = a * b

		return vars_coeff

	# Linear combination trait generator
	def lin_traits(pi_nm, co):
		return np.dot(pi_nm, co)

	# Quadratic combination trait generator
	def quad_traits(pi_nm, co):
		a = np.diag(np.random.randn(M))
		return np.dot(np.dot(np.dot(pi_nm, a), a.T),co)

	return snps, lin_traits(pi_nm, coeffs()), pi_nm

# snps, traits = gen_snps_traits()
# print(snps, traits)

import pandas as pd
def convert_to_df(snps, traits):
	individual_names = [('IND_' + str(x + 1)) for x in range(snps.shape[0])]
	df = pd.DataFrame(index=individual_names)
	for i in range(snps.shape[1]):
		ds = pd.Series(snps[:,i], index=individual_names, name=('SNP_'+str(i)))
		df = df.join(ds)
	df = df.join(pd.Series(traits, index=individual_names, name='Traits'))
	return df


data_type_params = {
	'small': {
		'logspace_snip_lo': 2,
		'logspace_snip_hi': 3,
		'num_snip_sets': 2,
		'linspace_inv_sparse_lo': 1,
		'linspace_inv_sparse_hi': 3,
		'num_sparse_sets': 3,
	},
	'medium': {
		'logspace_snip_lo': 3,
		'logspace_snip_hi': 4,
		'num_snip_sets': 2,
		'linspace_inv_sparse_lo': 10,
		'linspace_inv_sparse_hi': 70,
		'num_sparse_sets': 3,
	},
	'large': {
		'logspace_snip_lo': 5,
		'logspace_snip_hi': 6,
		'num_snip_sets': 2,
		'linspace_inv_sparse_lo': 100,
		'linspace_inv_sparse_hi': 500,
		'num_sparse_sets': 3,
	}

}
data_folder='SNPS_TRAITS_data'

from pathlib import Path
def create_data(data_type='small'):
	# create data, if not exist
	hdf_name = data_type + '.h5'
	hdf_file = Path(os.path.join(data_folder, hdf_name))
	if not (hdf_file.is_file()): # .h5 file does not exist
		for i, snip_size in enumerate(np.logspace(
				data_type_params[data_type]['logspace_snip_lo'],
				data_type_params[data_type]['logspace_snip_hi'],
				num=data_type_params[data_type]['num_snip_sets'])
		):
			for j, inv_sparse_factor in enumerate(np.linspace(
					data_type_params[data_type]['linspace_inv_sparse_lo'],
					data_type_params[data_type]['linspace_inv_sparse_hi'],
					num=data_type_params[data_type]['num_sparse_sets'])
			): # sparsity = 1./inv_sparse_factor
				individuals, snip_size = int(snip_size//inv_sparse_factor), int(snip_size)
				if individuals == 0:
					continue
				snips, traits, logits = gen_snps_traits(snip_size, individuals, 1./inv_sparse_factor)
				df = convert_to_df(snips, traits)
				key_value = ('data'+str(i)+'_'+str(j))
				df.to_hdf(hdf_file, key=key_value, mode='a')

				#logits
				df = convert_to_df(logits, traits)
				key_value = ('logits' + str(i) + '_' + str(j))
				df.to_hdf(hdf_file, key=key_value, mode='a')
	return

import random
def read_data(data_type='small'):
	hdf_name = data_type + '.h5'
	hdf_fname = os.path.join(data_folder, hdf_name)
	hdf_file_path = Path(hdf_fname)
	if not (hdf_file_path.is_file()): # .h5 file does not exist
		raise Exception("H5 file does not exist: " + hdf_file_path)
	hf = pd.HDFStore(hdf_fname, mode='r')
	data_keys = hf.keys()
	data_keys.sort()
	read_key = list(filter(lambda x: x.startswith('/data'), data_keys))[-1]
	df = pd.read_hdf(hdf_file_path, key=read_key, mode='r')

	nrows, ncols = df.shape
	all = random.sample(range(nrows), nrows)
	train_ix, test_ix  = all[:int(0.8*nrows)], all[int(0.8*nrows):]

	read_key = list(filter(lambda x: x.startswith('/logits'), data_keys))[-1]
	dl = pd.read_hdf(hdf_file_path, key=read_key, mode='r')

	# train, test (x_train, y_train, x_test, y_test)
	return df.ix[train_ix, :ncols-1], df.ix[train_ix, ncols-1], dl.ix[train_ix, :ncols-1],\
			df.ix[test_ix, :ncols-1], df.ix[test_ix, ncols-1], dl.ix[test_ix, ncols-1]

import os
def load_data(data_type='small'):
	data_folder = "SNPS_TRAITS_data"
	# Make dir, if not exist
	os.makedirs(data_folder, exist_ok=True)

	#data_types = ['small', 'medium', 'large']
	create_data(data_type)

	# Load data into a pd
	return read_data(data_type)

import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-D', "--data_type", help="small, medium or large", choices=['small', 'medium', 'large'], default='small', action='store')
	#parser.add_argument('-A', "--all", help="Generate all prebuilt synthetic datasets", default=True, action='store_true')
	#parser.add_argument('-M', "--snip_size", help="Number of genomic SNPs", default=30000)
	#parser.add_argument('-N', '--observations', help="Number of genomic observations", default=100)
	#parser.add_argument('-S', '--sparsity', help="Sparsity", default=0.3)
	#parser.add_argument('-R', '--relationship', help="Relationship b/w SNPs and traits", default='linear')

	args = parser.parse_args()
	(trainx, trainy, trainl, testx, testy, testl) = load_data(args.data_type)
	print(trainy, testy)
	# print(len(trainy), len(testy))
	# print(trainx, testx)
	# #logits
	# print(trainl, testl)

