import numpy as np
import pandas as pd
import os, argparse, numbers

# from tqdm import tqdm as tqdm

#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import tensorflow as tf

from .snps_traits import load_data

# from pandas_plink import read_plink
from dask.diagnostics import ProgressBar
# import dask, dask.array

# from framework.gwasData import *
# from framework.models import *
# import framework.regularizers as regularizers
#
# from multiprocessing.pool import ThreadPool

def l1reg(scale, scope=None):
	# boilerplate from tensorflow source code
	if isinstance(scale, numbers.Integral):
		raise ValueError('scale cannot be an integer: %s' % (scale,))
	if isinstance(scale, numbers.Real):
		if scale < 0.:
			raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
							 scale)
		if scale == 0.:
			tf.logging.info('Scale of 0 disables regularizer.')
			return lambda _: None

	# heart of regularizer
	def l1(weights):
		"""Applies l1 regularization to weights."""
		with tf.name_scope(scope, 'l1_regularizer', [weights]) as name:
			my_scale = tf.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
			return tf.multiply(my_scale, tf.reduce_sum(tf.abs(weights)), name=name)

	return l1

def linearRegression(features, labels, mode, params):
	# if params['regularizer'] == 'l1':
	# 	kernel_regularizer = l1reg(scale=params['lmbda'])
	# elif params['regularizer'] == 'l2':
	# 	kernel_regularizer = l2reg(scale=params['lmbda'])
	# else:
	# 	kernel_regularizer = None

	kernel_regularizer = l1reg(scale=params['lmbda'])

	# define model here
	output = tf.layers.dense(features[params['key']],
							 params['n_dims'],
							 activation=None,
							 name='weights',
							 use_bias=True,
							 kernel_regularizer=kernel_regularizer)

	# shortcut out if it's just a prediction
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'y_pred': output
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	# loss + regularization
	loss = tf.reduce_mean(tf.reduce_sum((labels - output) ** 2, axis=-1))

	if kernel_regularizer is not None:
		loss += tf.reduce_sum(tf.losses.get_regularization_losses())

	# evaluation metrics
	accuracy = tf.metrics.mean_squared_error(labels=labels, predictions=output, name='mse_op')

	# store metrics
	rSquared = 1 - tf.reduce_sum((output - labels) ** 2) / tf.reduce_sum((labels - tf.reduce_mean(labels)) ** 2)
	mRSquared, updateRSquared = tf.metrics.mean(rSquared)
	metrics = {'accuracy': accuracy, 'r_squared': (mRSquared, updateRSquared)}
	tf.summary.scalar('accuracy', accuracy[1])
	tf.summary.scalar('r_squared', rSquared)

	# different modes
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
	elif mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train regression')
	# parser.add_argument('--bfile', metavar='b', type=str, required=True, help='bed file')
	# parser.add_argument('--phenoFile', metavar='p', type=str, required=True, help='phenotype csv file')
	# parser.add_argument('--phenoColumn', metavar='c', type=str, required=True, help='phenotype column header')
	# parser.add_argument('--modelDir', metavar='m', type=str, required=True, help='Model directory')
	# parser.add_argument('--zarrFile', metavar='z', type=str, required=True, help='Zarr file with standardized inputs')
	# parser.add_argument('--alpha', metavar='a', type=float, required=False, default=1e-4, help='Learning rate')
	# parser.add_argument('--lmbda', metavar='l', type=float, required=True, help='Regularization parameter')
	# parser.add_argument('--numThreads', metavar='t', type=int, required=False, default=4,
	# 					help='Number of threads to use')
	# parser.add_argument('--epochs', metavar='l', type=int, required=False, default=20, help='Number of epochs')
	args = parser.parse_args()

	# # arguments
	numThreads = 1 # args.numThreads
	# ## dask options
	# dask.config.set(pool=ThreadPool(numThreads))
	## tensorflow options
	tf.logging.set_verbosity(tf.logging.INFO)
	## model options
	batchSize = 1024
	chunkSize = 1024
	nEpochs = 3 # args.epochs
	# alpha = args.alpha
	# lmbda = args.lmbda

	# print('Loading plink data matrix')
	# bim, fam, bed_T = read_plink(args.bfile)
	# bed = bed_T.transpose()

	nSteps = nEpochs * int(np.ceil(bed.shape[0] / batchSize))

	# print('Reading phenotype data')
	# phenoDf = pd.read_csv(args.phenoFile, index_col=0)
	# yData = phenoDf.loc[map(int, fam['iid'].values), :][args.phenoColumn].values.reshape(-1, 1).astype(np.float32)

	(trainx, trainy, trainl, trainz, trainw, testx, testy, testl, testz, testw) = load_data(args.data_type)

	runConfig = tf.estimator.RunConfig(
		session_config=tf.ConfigProto(
			intra_op_parallelism_threads=numThreads,
			inter_op_parallelism_threads=numThreads,
			allow_soft_placement=False,
			device_count={'CPU': numThreads}
		)
	)

	print('Initializing tensorflow estimator: Learning rate = {}, lambda = {}'.format(alpha, lmbda))
	# linear regression
	lr = tf.estimator.Estimator(
		model_fn=linearRegression,
		params=dict(n_dims=1, learning_rate=alpha, regularizer='l1', lmbda=lmbda, key='snps'),
		model_dir=args.modelDir, config=runConfig
	)

	print('Training model')
	lr.train(input_fn=lambda: makeLargeGenomicsDataset(args.zarrFile,
													   yData, chunkSize=chunkSize,
													   batchSize=batchSize),
			 steps=nSteps)

	# logistic regression
	#     lr = tf.estimator.Estimator(
	#         model_fn=logisticRegression,
	#         params=dict(n_dims=1, learning_rate=alpha, regularizer='l1', lmbda=lmbda, key='snps', n_classes=2),
	#         model_dir=args.modelDir, config=runConfig
	#     )

	#     lr.train(input_fn=lambda: makeLargeGenomicsDataset(args.zarrFile,
	#                                                        yData, chunkSize=chunkSize,
	#                                                        batchSize=batchSize, oneHot=2),
	#      steps=nSteps)