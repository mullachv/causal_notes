import numpy as np
import tensorflow as tf
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug

# Turn off interactive plotting
plt.ioff()


#
#
# https://jmetzen.github.io/2015-11-27/vae.html
#
#

np.random.seed(40997)
tf.set_random_seed(40997)

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

logdir = "./tfb_logs/"
tf.logging.set_verbosity(tf.logging.ERROR)

def xavier_init(fan_in, fan_out, constant=1):
	'''
	Initialize network weights
	:return:
	'''
	low = -constant*np.sqrt(6./(fan_in + fan_out))
	high = -low
	return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class VAE(object):
	def __init__(self, network_architecture, transfer_fn=tf.nn.softplus,
				 learning_rate=0.01, batch_size=100):
		self.network_architecture = network_architecture
		self.transfer_fn = transfer_fn
		self.learning_rate = learning_rate
		self.batch_size = batch_size


		with tf.name_scope("variables_scope"):
			self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

		self._create_network()
		self._create_loss_optimizer()
		init = tf.global_variables_initializer()

		#Launch session
		self.session = tf.InteractiveSession()
		self.session.run(init)

		# To debug using tensorboard
		# 	sess = tf.Session()
		# 	self.session = tf_debug.TensorBoardDebugWrapperSession(sess, "127.0.0.1:8080")
		# 	self.session.run(init)
		#

	def _create_network(self):
		network_weights = self._initialize_weights(**self.network_architecture)
		self.z_mean, self.z_log_sigma_sq = \
			self._recognition_network(network_weights["weights_recog"], network_weights["biases_recog"])

		#draw a sample from Gaussian distr.
		n_z = self.network_architecture["n_z"]
		eps = tf.random_normal((self.batch_size, n_z), 0,1,dtype=tf.float32)

		#z = mu + sigma*eps
		self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

		self.x_reconstruct_mean = self._generator_network(network_weights["weights_gener"],
														  network_weights["biases_gener"])

	def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, n_hidden_gener_1, n_hidden_gener_2,
							n_input, n_z):
		with tf.name_scope("weights_scope"):
			all_weights = dict()
			all_weights['weights_recog'] = {
				'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
				'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
				'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
				'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))
			}
			all_weights['biases_recog'] = {
				'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
				'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
				'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
				'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))
			}
			all_weights['weights_gener'] = {
				'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
				'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
				'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
				'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_input))
			}
			all_weights['biases_gener'] = {
				'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
				'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
				'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
				'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))
			}
			# tf.summary.histogram("recog", all_weights['weights_recog'])
			# tf.summary.histogram("gener", all_weights['weights_gener'])

		return all_weights

	def _recognition_network(self, weights, biases):
		with tf.name_scope("recog_layers"):
			layer_1 = self.transfer_fn(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
			layer_2 = self.transfer_fn(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
			z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
			z_sigma_log_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])

		return (z_mean, z_sigma_log_sq)

	def _generator_network(self, weights, biases):
		layer_1 = self.transfer_fn(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
		layer_2 = self.transfer_fn(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
		x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean']))
		return x_reconstr_mean

	def _create_loss_optimizer(self):
		# The loss is composed of two terms:
		# 1.) The reconstruction loss (the negative log probability
		#     of the input under the reconstructed Bernoulli distribution
		#     induced by the decoder in the data space).
		#     This can be interpreted as the number of "nats" required
		#     for reconstructing the input when the activation in latent
		#     is given.
		# Adding 1e-10 to avoid evaluation of log(0.0)
		reconstr_loss = \
			-tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstruct_mean) +
								(1-self.x)*tf.log(1e-10 + 1-self.x_reconstruct_mean), 1)

		# 2.) The latent loss, which is defined as the Kullback Leibler divergence
		##    between the distribution in latent space induced by the encoder on
		#     the data and some prior. This acts as a kind of regularizer.
		#     This can be interpreted as the number of "nats" required
		#     for transmitting the the latent space distribution given
		#     the prior.
		latent_loss = -0.5*tf.reduce_sum(1+self.z_log_sigma_sq - tf.square(self.z_mean)
										 -tf.exp(self.z_log_sigma_sq), 1)
		#avg over batch
		with tf.name_scope("loss_scope"):
			self.cost = tf.reduce_mean(reconstr_loss + latent_loss, name='loss')

		with tf.name_scope("training_scope"):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
			tf.summary.histogram("loss_histogram", self.cost)
			tf.summary.scalar("loss_scalar", self.cost)

	def partial_fit(self, X):
		#Return cost of mini-batch
		opt, cost = self.session.run((self.optimizer, self.cost), feed_dict={self.x: X})
		return cost

	def transform(self, X):
		"""Transform data by mapping it into the latent space."""
		# Note: This maps to mean of distribution, we could alternatively
		# sample from Gaussian distribution
		return self.session.run(self.z_mean, feed_dict={self.x: X})

	def generate(self, z_mu=None):
		"""
		Generate data by sampling from latent space.
		If z_mu is not None, data for this point in latent space is
		generated. Otherwise, z_mu is drawn from prior in latent space.
		"""
		if z_mu is None:
			z_mu = np.random.normal(size=self.network_architecture['n_z'])

		return self.session.run(self.x_reconstruct_mean, feed_dict={self.z: z_mu})

	def reconstruct(self, X):
		""" Use VAE to reconstruct given data. """
		return self.session.run(self.x_reconstruct_mean, feed_dict={self.x: X})

	def cleanup(self):
		self.session.close()

def train(network_architecture, learning_rate=0.001, batch_size=100,
		  training_epochs=10, display_steps=5):
	vae = VAE(network_architecture, transfer_fn=tf.nn.softplus, learning_rate=learning_rate, batch_size=batch_size)
	merged_summary_op = tf.summary.merge_all()
	tb_writer = tf.summary.FileWriter(logdir)
	tb_writer.add_graph(vae.session.graph)

	saver = tf.train.Saver()

	for epoch in range(training_epochs):
		avg_cost = 0
		nbatches = int(n_samples / batch_size)

		#Loop over all batches
		for i in range(nbatches):
			batch_xs, _ = mnist.train.next_batch(batch_size)

			#fit training using batch data
			cost = vae.partial_fit(batch_xs)
			avg_cost += cost * batch_size / n_samples

		if epoch % display_steps == 0:
			print("Epoch: {:04d}, cost={}".format(epoch+1, avg_cost))

	saver.save(vae.session, './vae_model.ckpt')
	print("Saved VAE model")
	return vae

network_architecture = \
	dict(
		n_hidden_recog_1 = 200,
		n_hidden_recog_2 = 200,
		n_hidden_gener_1 = 200,
		n_hidden_gener_2 = 200,
		n_input = 784,
		n_z = 20
		 )
vae = train(network_architecture, training_epochs=4, display_steps=1)
x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)
fig = plt.figure(figsize=(8, 12))
for i in range(5):
	plt.subplot(5, 2, 2*i + 1)
	plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap='gray')
	plt.title('Test input {}'.format(i+1))
	plt.colorbar()
	plt.subplot(5,2,2*i+2)
	plt.imshow(x_reconstruct[i].reshape(28,28), vmin=0, vmax=1, cmap='gray')
	plt.title('Reconstructed')
	plt.colorbar()

plt.tight_layout()
plt.savefig('./reconstruct.png')
plt.close(fig)

#
# Plot 1st two coordinates of the latent space
#
#vae_2d = train(network_architecture, training_epochs=4, display_steps=1)
x_sample, y_sample = mnist.test.next_batch(5000)
z_mu = vae.transform(x_sample)
fig = plt.figure(figsize=(8,6))
plt.scatter(z_mu[:,0], z_mu[:,1],c=np.argmax(y_sample, 1))
plt.colorbar()
plt.grid()
plt.savefig('./latent_space.png')
plt.close(fig)

#
# Generate images from vectors of length 20, within [-3, 3]
#
nx = ny = 20
x_values = np.linspace(-3,3,20)
y_values = np.linspace(-3,3,20)
canvas = np.empty((28*ny, 28*nx))
for i, yi in enumerate(x_values):
	for j, xi in enumerate(y_values):
		z_mu = np.array([[xi, yi]*int(nx/2)]*vae.batch_size)
		x_mean = vae.generate(z_mu)
		canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28,28)

fig = plt.figure(figsize=(8,10))
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.tight_layout()
plt.savefig('./digit_canvas.png')
plt.close()

vae.cleanup()
print("Cleaned up session run")
