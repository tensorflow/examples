import glob
import json



from data_pipeline import input_fn
from gpt2_model import *

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/model"


#默认训练参数
#num_layers = 8
#embedding_size = 768
#num_heads = 8
#dff =3072
#max_seq_len = 515 
#vocab_size = 24512
#optimizer = "adam"
#batch_size = 8
#learning_rate = 0.001
#graph_mode = False
#distributed = False




#编写训练函数
def train(num_layers = 8, embedding_size = 768, num_heads = 8, dff = 3072, max_seq_len = 515, vocab_size = 24512,
          optimizer = "adam", batch_size = 8, learning_rate = 0.001, graph_mode = False, distributed = False):
	par_map = {"num_layers": num_layers, "d_model": embedding_size,
	           "num_heads": num_heads, "dff": dff,
	           "max_seq_len": max_seq_len, "vocab_size": vocab_size}

	# exp_name = "_".join(['{}_{}'.format(k, v) for k, v in par_map.items()])

	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)

	with open(MODEL_DIR + '/model_par.json', 'w') as f:
		json.dump(par_map, f)

	tf_records = glob.glob((_ROOT + "/data/tf_records/*.tfrecord"))
	train_percent = int(len(tf_records) * (85 / 100))

	print("No. of tf records:- ", len(tf_records))
	train_tf_records = tf_records[:train_percent]
	test_tf_records = tf_records[train_percent:]

	train_dataset = input_fn(train_tf_records, batch_size=batch_size)
	test_dataset = input_fn(test_tf_records, batch_size=batch_size)

	if distributed:
		mirrored_strategy = tf.distribute.MirroredStrategy()
		train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
		test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)

		
		
		with mirrored_strategy.scope():
			#定义模型
			model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
			             optimizer=optimizer, learning_rate=learning_rate)
			model.create_optimizer()
			model.create_checkpoint_manager(MODEL_DIR)
			model.create_summary_writer(LOG_DIR)

			
		model.mirrored_strategy = mirrored_strategy
		model.global_batch_size = tf.cast(batch_size, tf.float32)
		
		
		
		
		
		
	else:
		model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
		             optimizer=optimizer, learning_rate=learning_rate)
		model.create_optimizer()
		model.create_checkpoint_manager(MODEL_DIR)
		model.create_summary_writer(LOG_DIR)

	model.fit([train_dataset, test_dataset], graph_mode)# graph_mode对于GPU可以为true
	print("===============>>>>>>>>>>>>>>>Done!11")


if __name__ == "__main__":
	train()
	#__name__ 是当前模块名，当模块被直接运行时模块名为 __main__ 。这句话的意思就是，当模块被直接运行时，以下代码块将被运行，当模块是被导入时，代码块不被运行。
