from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(40997)

from snps_traits import gen_snps_traits

#Factor Analyses
def fas():
	methods = {
		'FA': decomposition.FactorAnalysis(),
		'ICA': decomposition.FastICA(),
		'LDA': decomposition.LatentDirichletAllocation(),
		'PCA': decomposition.PCA(n_components=10),
		'NMF': decomposition.NMF(),
		#'RPCA': decomposition.RandomizedPCA(n_components=K)
	 }

	def m_analysis(matr, which='PCA'):
		'''
		Factorize as specified
		:param matr:
		:return: w, z such that w * z = matr
		'''
		matr = StandardScaler().fit_transform(matr)
		#to avoid evaluating log(0.0)
		matr += 1e-10
		print(matr)
		return methods[which].fit_transform(matr)

	snps, traits = gen_snps_traits(100, 5, 2)
	print(snps, traits)

	# print(m_analysis(snps, 'FA'))
	# print(m_analysis(snps, 'PCA'))
	# print(m_analysis(snps, 'ICA'))

fas()