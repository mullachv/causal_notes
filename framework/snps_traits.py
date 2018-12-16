import numpy as np
from scipy.stats import binom

# M number of genomic features, around 100,000 to 1 million,
# 	each feature = 0,1,2  (binomial)
# N number of individuals, around 5000
# M ~ 200 * N
M = 5
N = 2
K = 2

np.random.seed(40997)

# Generate Synthetic SNPs and Traits
def gen_snps_traits():
	def gen_pi_nm():
		return np.random.rand(N, M)

	def gen_xnm(pi_nm):
		return binom.rvs(2, pi_nm)

	pi_nm = gen_pi_nm()
	snps = gen_xnm(pi_nm)

	# print(snps)
	# print(pi_nm)

	vars_coeff_ind, vars_coeff = 0, None

	#What fraction of M's contribute to y_n
	CMP_FACTOR = 0.3

	def coeffs():
		nonlocal vars_coeff_ind, vars_coeff
		if vars_coeff_ind == 0:
			vars_coeff_ind = 1
			a = binom.rvs(1, [CMP_FACTOR]*pi_nm.shape[1])
			b = np.random.randn(pi_nm.shape[1])
			vars_coeff = a * b

		return vars_coeff

	#print(coeffs())

	#Linear combination trait generator
	def lin_traits(pi_nm, co):
		return np.dot(pi_nm, co)

	def quad_traits(pi_nm, co):
		a = np.diag(np.random.randn(M))
		#print(a)
		return np.dot(np.dot(np.dot(pi_nm, a), a.T),co)

	#print(quad_traits(pi_nm, coeffs()))

	return snps, lin_traits(pi_nm, coeffs())

snps, traits = gen_snps_traits()
print(snps, traits)

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

#Factor Analyses
def fas():
	methods = {
		'FA': decomposition.FactorAnalysis(),
		'ICA': decomposition.FastICA(),
		'LDA': decomposition.LatentDirichletAllocation(),
		'PCA': decomposition.PCA(n_components=K),
		'NMF': decomposition.NMF(),
		#'RPCA': decomposition.RandomizedPCA(n_components=K)
	 }

	def m_analysis(matr, which='RPCA'):
		'''
		Factorize as specified
		:param matr:
		:return: w, z such that w * z = matr
		'''
		matr = StandardScaler().fit_transform(matr)
		return methods[which].fit_transform(matr)

	print(m_analysis(snps, 'FA'))
	print(m_analysis(snps, 'PCA'))

fas()
