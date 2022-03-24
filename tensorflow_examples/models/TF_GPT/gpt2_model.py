"""
模型文件与layers和utils关联
"""

import os
from tensorflow.python.framework import tensor_shape
from layers.attention_layer import *
from layers.embedding_layer import *
from layers.feed_forward import *
from layers.layer_norm import LayerNormalization
from utils.tf_utils import *





_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"

train_step_signature = [
	tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Inputs"),
	tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Targets")
]


class Gpt2(tf.keras.Model):
	def __init__(self, num_layers,
	             d_model,
	             num_heads,
	             dff,
	             max_seq_len,
	             vocab_size,
	             optimizer="adam",
	             learning_rate=1e-3,
	             rev_embedding_projection=True,
	             grad_clip=False,
	             clip_value=1.0):
		super(Gpt2, self).__init__()

		self.rev_embedding_projection = rev_embedding_projection
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.dff = dff
		self.max_seq_len = max_seq_len
		self.vocab_size = vocab_size
		self.d_model = d_model
		self.learning_rate = learning_rate
		self.optimizer_t = optimizer
		self.mirrored_strategy = None
		self.grad_clip = grad_clip
		self.clip_value = clip_value

		self.embedding = EmbeddingLayer(
			self.vocab_size, self.d_model)

		self.pos_embedding = PositionEmbeddingLayer(
			self.max_seq_len, self.d_model)

		self.decoder_layers = [DecoderLayer(self.d_model, self.num_heads, self.dff)
		                       for _ in range(self.num_layers)]
		self.layer_norm = LayerNormalization(self.d_model)

		if not self.rev_embedding_projection:
			self.output_layer = OutputLayer(self.vocab_size)

		self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
			from_logits=True, reduction='none')

		self.accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy(
			name='accuracy')

		self.train_step_signature = [
			tf.TensorSpec(shape=(None, None), dtype=tf.int32)]

	def call(self, x, training=True, past=None):
		x = tf.cast(x, tf.int32)
		# self.batch_size, self.sequence = tf.shape(x)[0], tf.shape(x)[1]
		if past is None:
			pasts = [None] * self.num_layers
		else:
			pasts = past

		assert len(pasts) == self.num_layers

		att_mask = create_masks(x)
		past_length = 1 if past is None else tf.shape(past)[-2]
		with tf.name_scope("embeddings"):
			embedded_x = self.embedding(x)
			hidden_states = embedded_x + self.pos_embedding(x, start=past_length)

		presents = []
		for decoder_layer, past in zip(self.decoder_layers, pasts):
			hidden_states, present = decoder_layer(hidden_states, training, att_mask, past=past)
			presents.append(present)

		hidden_states = self.layer_norm(hidden_states)

		if self.rev_embedding_projection:
			logits = self.embedding(hidden_states, mode="projection")
		else:
			logits = self.output_layer(hidden_states)

		return logits, presents

	@staticmethod
	def get_padded_accuracy(labels, logits):
		with tf.name_scope("padded_accuracy"):
			weights = tf.cast(tf.not_equal(labels, 0), tf.float32)

			outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
			padded_labels = tf.cast(labels, tf.int32)

			nonpad_seq = tf.math.count_nonzero(weights, dtype=tf.dtypes.float32, )
			acc = tf.cast(tf.equal(outputs, padded_labels), tf.float32)

			accuracy = tf.reduce_sum(tf.cast(acc * weights, tf.float32)) / nonpad_seq
			return tf.cast(accuracy, tf.float32)

	def create_optimizer(self):
		optimizer = self.optimizer_t.lower()
		with tf.name_scope("optimizer"):
			if optimizer == "adam":
				self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,
				                                          epsilon=1e-9)
			elif optimizer == "adadelta":
				self.optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)
			elif optimizer == "rms":
				self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
			else:
				self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
			return self.optimizer

	def get_loss(self, real, pred):
		with tf.name_scope("loss_layer"):
			mask = tf.math.logical_not(tf.math.equal(real, 0))
			loss_ = self.loss_object(real, pred)

			with tf.name_scope("loss_masking"):
				mask = tf.cast(mask, dtype=loss_.dtype)
				loss_ *= mask
			loss_ = tf.reduce_sum(loss_, axis=1)
			sequence_avg_loss = loss_ / tf.reduce_sum(mask, axis=1)
			return sequence_avg_loss

	@staticmethod
	def get_perplexity(cross_entropy):
		perplexity = tf.exp(cross_entropy)
		return perplexity

	def create_checkpoint_manager(self, checkpoint_path, max_to_keep=5, load_model=True):
		with tf.name_scope('checkpoint_manager'):
			ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
			self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

			if load_model:  # If want to load trained weights
				ckpt.restore(self.ckpt_manager.latest_checkpoint)
				print('Latest checkpoint restored...............')
			else:
				print("Initializing model from scratch..........")

	def load_model(self, filepath):
		ckpt = tf.train.Checkpoint(model=self)
		ckpt_manager = tf.train.CheckpointManager(ckpt, filepath)
		ckpt.restore(ckpt_manager.latest_checkpoint)
		print("Model Restored..........................")

	def create_summary_writer(self, summary_path):
		train_summary_path = summary_path + "/train"
		test_summary_path = summary_path + "/test"

		with tf.name_scope('summary'):
			self.train_writer = tf.summary.create_file_writer(train_summary_path)
			self.test_writer = tf.summary.create_file_writer(test_summary_path)

			return self.train_writer, self.test_writer

	def _train_step(self, inputs, targets):
		with tf.GradientTape() as tape:
			predictions, _ = self(inputs, training=True)
			loss = tf.reduce_mean(self.get_loss(targets, predictions))

		with tf.name_scope("gradients"):
			gradients = tape.gradient(loss, self.trainable_variables)
			if self.grad_clip:
				gradients = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value))
				             for grad in gradients]
			self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		perplexity = self.get_perplexity(loss)
		step = self.optimizer.iterations

		return step, loss, perplexity

	def _test_step(self, inputs, targets):
		pred, _ = self(inputs, training=False)
		loss = self.get_loss(targets, pred)
		perplexity = self.get_perplexity(loss)
		return loss, perplexity

	@tf.function(input_signature=train_step_signature)
	def train_step(self, inputs, targets):
		return self._train_step(inputs, targets)

	@tf.function(input_signature=train_step_signature)
	def test_step(self, inputs, targets):
		return self._test_step(inputs, targets)

	def _distributed_train_step(self, inputs, targets):

		def step_fn(inp, tar):
			with tf.GradientTape() as tape:
				logits, _ = self(inp, training=True)
				cross_entropy = self.get_loss(tar, logits)
				loss = tf.reduce_sum(cross_entropy) * (1.0 / self.global_batch_size)  # Divided By Global Batch Size

			with tf.name_scope("gradients"):
				gradients = tape.gradient(loss, self.trainable_variables)
				if self.grad_clip:
					gradients = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value))
					             for grad in gradients]
				self.optimizer.apply_gradients(list(zip(gradients, self.trainable_variables)))
			return cross_entropy

		per_example_losses = self.mirrored_strategy.run(
			step_fn, args=(inputs, targets))

		mean_loss = self.mirrored_strategy.reduce(
			tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
		# If you get error in distributed mode try using SUM instead of MEAN.

		perplexity = self.get_perplexity(mean_loss)
		step = self.optimizer.iterations

		return step, mean_loss, perplexity

	def _distributed_test_step(self, inputs, targets):
		def step_fn(inp, tar):
			logits, _ = self(inp, training=False)
			cross_entropy = self.get_loss(tar, logits)
			return cross_entropy

		per_example_losses = self.mirrored_strategy.run(
			step_fn, args=(inputs, targets))

		mean_loss = self.mirrored_strategy.reduce(
			tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
		# If you get error in distributed mode try using SUM instead of MEAN.
		perplexity = self.get_perplexity(mean_loss)

		return mean_loss, perplexity

	@tf.function(experimental_relax_shapes=True)
	def distributed_train_step(self, inputs, targets):
		return self._distributed_train_step(inputs, targets)

	@tf.function(experimental_relax_shapes=True)
	def distributed_test_step(self, inputs, targets):
		return self._distributed_test_step(inputs, targets)

	def get_train_test_function(self, graph_mode=False):
		if graph_mode:
			print("Running in graph mode.............")
			train_fuc = self.train_step
			test_fuc = self.test_step
		else:
			print("Running in eager mode.............")
			train_fuc = self._train_step
			test_fuc = self._test_step
		return train_fuc, test_fuc

	def get_distributed_train_test_function(self, graph_mode=False):
		if graph_mode:
			print("Running in graph mode.............")
			train_fuc = self.distributed_train_step
			test_fuc = self.distributed_test_step
		else:
			print("Running in eager mode.............")
			train_fuc = self._distributed_train_step
			test_fuc = self._distributed_test_step
		return train_fuc, test_fuc

	def fit(self, train_dataset, graph_mode):
		if self.mirrored_strategy is None:
			train_dataset, test_dataset = train_dataset
			train_func, test_func = self.get_train_test_function(graph_mode)
			tf.summary.trace_on(graph=True, profiler=False)
			for (_, (inputs, targets)) in enumerate(train_dataset):
				step, loss, perplexity = train_func(inputs, targets)
				if step % 100 == 0:
					self.log_summary(self.train_writer,
					                 step.numpy(),
					                 loss.numpy(),
					                 perplexity.numpy())

				if step == 0:
					with self.train_writer.as_default():
						tf.summary.trace_export(
							name="gpt-2",
							step=0,
							profiler_outdir=LOG_DIR)

				if step % 500 == 0:
					losses = []
					perplexities = []
					for (test_step, (test_inputs, test_targets)) in enumerate(test_dataset):
						test_loss, test_perplexity = test_func(test_inputs, test_targets)
						losses.append(test_loss)
						perplexities.append(test_perplexity)

						if test_step == 100:
							break

					test_loss = np.mean(np.array(losses))
					test_perplexity = np.mean(np.array(perplexities))

					self.log_summary(self.test_writer,
					                 step.numpy(),
					                 test_loss,
					                 test_perplexity,
					                 result_type="Test")

					ckpt_save_path = self.ckpt_manager.save()
					print('Saving checkpoint for step {} at {}'.format(step.numpy(),
					                                                   ckpt_save_path))
		else:
			with self.mirrored_strategy.scope():
				train_dataset, test_dataset = train_dataset
				train_func, test_func = self.get_distributed_train_test_function(graph_mode)
				tf.summary.trace_on(graph=True, profiler=False)
				for (step, (inputs, targets)) in enumerate(train_dataset):
					step, loss, perplexity = train_func(inputs, targets)

					if step % 100 == 0:
						self.log_summary(self.train_writer,
						                 step,
						                 loss,
						                 perplexity)

					if step == 0:
						with self.train_writer.as_default():
							tf.summary.trace_export(
								name="gpt-2",
								step=0,
								profiler_outdir=LOG_DIR)

					if step % 500 == 0:
						losses = []
						perplexities = []
						for (test_step, (test_inputs, test_targets)) in enumerate(test_dataset):
							test_loss, test_perplexity = test_func(test_inputs, test_targets)
							losses.append(test_loss)
							perplexities.append(test_perplexity)

							if test_step == 100:
								break

						test_loss = np.mean(np.array(losses))
						test_perplexity = np.mean(np.array(perplexities))

						self.log_summary(self.test_writer,
						                 step,
						                 test_loss,
						                 test_perplexity,
						                 result_type="Test")

						ckpt_save_path = self.ckpt_manager.save()
						print('Saving checkpoint for step {} at {}'.format(step.numpy(),
						                                                   ckpt_save_path))

	@staticmethod
	def log_summary(tf_writer, step, loss, perplexity, result_type="Train"):
		print(result_type + ':- Step {}, Loss {:.4f}, Perplexity {:.4f}'.format(
			step, loss, perplexity))
		with tf_writer.as_default():
			tf.summary.scalar("loss", loss, step=step)
			tf.summary.scalar("perplexity", perplexity, step=step)


class OutputLayer(tf.keras.layers.Layer):
	def __init__(self, output_dim, proj_weights=None, kernel_initializer=None):
		super(OutputLayer, self).__init__()
		self.proj_weights = proj_weights
		self.output_dim = output_dim
		self.layer_weights = None
		self.kernel_initializer = kernel_initializer

	def build(self, input_shape):
		if self.proj_weights is None:
			input_dim = tensor_shape.dimension_value(input_shape[-1])
			self.layer_weights = self.add_weight(
				'output_layer_weights',
				shape=[input_dim, self.output_dim],
				initializer=self.kernel_initializer,
				trainable=True)
		super(OutputLayer, self).build(input_shape)

	def call(self, x):
		batch, sequence, d_model = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[-1]
		h_flat = tf.reshape(x, [-1, d_model])

		if self.proj_weights is None:
			out = tf.matmul(h_flat, self.layer_weights)
		else:
			out = tf.matmul(h_flat, self.porj_weights, transpose_b=True)
		out = tf.reshape(out, [batch, sequence, self.output_dim])
		return out


class DecoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff,
	             dr_rate=0.1):
		super(DecoderLayer, self).__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.dff = dff
		self.dr_rate = dr_rate

		self.mha = MultiHeadAttention(self.d_model, self.num_heads)
		self.feed_forward = FeedForward(self.d_model, self.dff, self.dr_rate)
		self.layer_norm1 = LayerNormalization(self.d_model)
		self.layer_norm2 = LayerNormalization(self.d_model)

	def call(self, x, training, mask, past=None):
		out, present = self.mha(self.layer_norm1(x), mask=mask, past_layer=past,
		                        training=training)  # (batch_size, input_seq_len, d_model)
		with tf.name_scope("residual_conn"):
			x = x + out
		out = self.feed_forward(self.layer_norm2(x), training=training)  # (batch_size, input_seq_len, d_model)
		with tf.name_scope("residual_conn"):
			x = x + out
		return x, present
