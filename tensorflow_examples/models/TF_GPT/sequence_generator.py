# this file was gener the sequences
from sample import SequenceGenerator
import click


@click.command()
@click.option('--model-path', type=str, default="./model", show_default=True, help="Model Path")
@click.option('--model-param', type=str, default="./model/model_par.json", show_default=True, help="Model Parm")
@click.option('--vocab', type=str, default="./data/bpe_model.model", show_default=True, help="Vocab")
@click.option('--seq-len', type=int, default=512, show_default=True, help="seq_len")
@click.option('--temperature', type=float, default=1.0, show_default=True, help="seq_len")
@click.option('--top-k', type=int, default=8, show_default=True, help="seq_len")
@click.option('--top-p', type=float, default=0.9, show_default=True, help="seq_len")
@click.option('--nucleus_sampling', type=bool, default=False, show_default=True, help="seq_len")
@click.option('--context', type=str, default="sample context", show_default=True, help="Context given to model")


def sequence_gen(model_path, model_param, vocab, seq_len, temperature, top_k, top_p, nucleus_sampling, context):
	sg = SequenceGenerator(model_path, model_param, vocab)
	sg.load_weights()
	generated_seq = sg.sample_sequence(context,
									   seq_len=seq_len,
									   temperature=temperature,
									   top_k=top_k,
									   top_p=top_p,
									   nucleus_sampling=nucleus_sampling)
	print("<<<<<<<<<===================================Sample===================================>>>>>>>>>>>>>>\n\n " + generated_seq)


if __name__ == "__main__":
	sequence_gen()#生成句子
