import numpy as np
np.random.seed(40997)

#
#  With Tensorflow backend
#
from tensorflow import set_random_seed
set_random_seed(40997)

#
# https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
#
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
class roc_callback(Callback):
	def __init__(self,training_data,validation_data, store_metrics, store_key):
		self.x = training_data[0]
		self.y = training_data[1]
		self.x_val = validation_data[0]
		self.y_val = validation_data[1]
		try:
			store_metrics[store_key]
		except:
			store_metrics[store_key] = []
		self.store_metrics = store_metrics[store_key]

	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		y_pred = self.model.predict(self.x)
		roc = roc_auc_score(self.y, y_pred)
		y_pred_val = self.model.predict(self.x_val)
		roc_val = roc_auc_score(self.y_val, y_pred_val)
		self.store_metrics += (roc, roc_val),
		#print('\rroc-auc: {:.4f} - roc-auc_val: {:.4f}\n'.format(roc, roc_val))
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return


def bin_xentry(ypred, ytrue):
	return np.sum(ytrue * np.log(ypred) + (1. - ytrue)*np.log(1.-ypred))

from keras.layers import Dense, Input
from keras.models import Model
from keras import regularizers

from synthetic_data import load_training_data, load_validation_data, load_test_data
tx, _, ty, _ = load_training_data()
vx, _, vy, _ = load_validation_data()
sx, _, sy, _ = load_test_data()

# tx = np.squeeze(tx, axis=0)
# ty = np.squeeze(ty, axis=0)
# vx = np.squeeze(vx, axis=0)
# vy = np.squeeze(vy, axis=0)
# sx = np.squeeze(sx, axis=0)
# sy = np.squeeze(sy, axis=0)
#

margs = {
	'tx': np.squeeze(tx, axis=0),
	'ty': np.squeeze(ty, axis=0),
	'vx': np.squeeze(vx, axis=0),
	'vy': np.squeeze(vy, axis=0),
	'sx': np.squeeze(sx, axis=0),
	'sy': np.squeeze(sy, axis=0),
	'l1': 'l1',
	'reg': .02,
	'store_key': 'Logistic'
}
store_metrics = {}
def _logistic(tx, ty, vx, vy, l1='l1', reg=.02, store_key='Logistic'):
	a = Input(shape=(tx.shape[1],))
	kr = regularizers.l1(reg) if l1 == 'l1' else regularizers.l2(reg)
	output = Dense(1, activation='sigmoid', kernel_regularizer=kr)(a)
	logistic = Model(inputs=a, outputs=output)

	logistic.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',])
	logistic.fit(x=tx, y=ty, validation_data=(vx, vy),
				 epochs=30,
				 batch_size=10,
				 shuffle=True,
				 callbacks=[roc_callback(training_data=(tx, ty),
										 validation_data=(vx, vy),
										 store_metrics=store_metrics,
										 store_key=store_key)],
				 verbose=0)
	# logistic.save('logistic_regr.h5')

def simple_logistic(**kwargs):
	return _logistic(kwargs['tx'],
					 kwargs['ty'],
					 kwargs['vx'],
					 kwargs['vy'],
					 'l1',
					 0,
					 kwargs['store_key'])

def logistic_with_l1(**kwargs):
	return _logistic(kwargs['tx'],
					 kwargs['ty'],
					 kwargs['vx'],
					 kwargs['vy'],
					 'l1',
					 kwargs['reg'],
					 kwargs['store_key'])

def logistic_with_l2(**kwargs):
	return _logistic(kwargs['tx'],
					 kwargs['ty'],
					 kwargs['vx'],
					 kwargs['vy'],
					 'l2',
					 kwargs['reg'],
					 kwargs['store_key'])

def logistic_on_princ_comps(**kwargs):
	U, _, _ = np.linalg.svd(kwargs['tx'], full_matrices=False)
	T, _, _ = np.linalg.svd(kwargs['vx'], full_matrices=False)
	return _logistic(U, kwargs['ty'], T, kwargs['vy'], 'l1', kwargs['reg'], kwargs['store_key'])

scenarios = {'No Reg' : [simple_logistic, 'l1', 0],
			'L1+.02' : [logistic_with_l1, 'l1', .02],
			'L1+.9' : [logistic_with_l1, 'l1', .9],
			'L2 .02' : [logistic_with_l2, 'l2', .02],
			'L1 100' : [logistic_with_l1, 'l1', 100],
			'PCA' :  [logistic_on_princ_comps, 'l1', 0],
			'PCA L1 .9': [logistic_on_princ_comps, 'l1', .9],
			'PCA L1 10' : [logistic_on_princ_comps, 'l1', 10],
			'PCA L1 100' : [logistic_on_princ_comps, 'l1', 100],
			'PCA L1 1000' :  [logistic_on_princ_comps, 'l1', 1000],
			'PCA L1 10000' : [logistic_on_princ_comps, 'l1', 10000],
			'PCA L1 100000' : [logistic_on_princ_comps, 'l1', 100000]
		}


def pick_best():
	# dd = np.load('store_metrics.npy')
	# u = dd.item()
	u = store_metrics
	for k, v in u.items():
		c, d = 0, 0
		for a, b in v:
			if b > c:
				c = b
				d = a
		print(k, c)

import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-D', "--data_root", help="Root data folder", default='./', type=str)
	parser.add_argument('-n', "--num_reps", help="Repetitions", default=2, type=int)
	args = parser.parse_args()

	for k in scenarios.keys():
		f = scenarios[k][0]
		margs['store_key'] = k
		margs['l1'] = scenarios[k][1]
		margs['reg'] = scenarios[k][2]
		[f(**margs) for _ in range(args.num_reps)]

	np.save(args.data_root + "store_metrics.npy", store_metrics)
	pick_best()
