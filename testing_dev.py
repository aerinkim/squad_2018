import re
import os
import sys
import random
import string
import logging
import argparse
import json
import torch
import pickle
import spacy
import tqdm
import numpy as np
from os.path import basename, dirname
from datetime import datetime
from collections import Counter, defaultdict
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGen
from my_utils.utils import set_environment
from my_utils.tokenizer import reform_text, Vocabulary, END
from my_utils.log_wrapper import create_logger
from my_utils.word2vec_utils import load_glove_vocab, build_embedding
#from my_utils.ensemble_utils import ensemble_avg, ensemble_count
from my_utils.squad_eval_v2 import *


my_meta = 'data_v2/squad_meta_v2.pick'
my_covec = 'data_v2/MT-LSTM.pt'
#elmo_options_path = 'data_resource/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
#elmo_weight_path = 'data_resource/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
glv = 'glove.840B.300d.updated.txt'

num_model = 20
have_gpu = True
test_file = 'data_v2/dev-v2.0.json'
#output_file = sys.argv[2]
#na_prob_file = sys.argv[3]
batch_size = 32
max_len = 5
num_tune = 89571
avg_on = False
do_count = True

logger = create_logger(__name__, to_disk=False)
workspace = '/home/aerin/Desktop/squad_vteam/'
model_root = workspace
glove_path = os.path.join(workspace, glv)
glove_dim = 300
meta_path = os.path.join(workspace, my_meta)
mtlstm_path = os.path.join(workspace, my_covec)
n_threads = 16

pad='-' * 10
logger.info('{}Resource Path{}'.format(pad, pad))
logger.info('workspace:{}'.format(workspace))
logger.info('model path:{}'.format(model_root))
logger.info('test file:{}'.format(test_file))
#logger.info('output file:{}'.format(output_file))
#logger.info('no answer prob file:{}'.format(na_prob_file))
logger.info('glove file:{}'.format(glove_path))
logger.info('meta file:{}'.format(meta_path))
logger.info('mtlstm file:{}'.format(mtlstm_path))
logger.info('processing data ...')





def load_data(path):
	rows = []
	with open(path, encoding="utf8") as f:
		data = json.load(f)['data']
	for article in tqdm.tqdm(data, total=len(data)):
		for paragraph in article['paragraphs']:
			context = paragraph['context']
			context = '{} {}'.format(context, END)
			for qa in paragraph['qas']:
				uid, question = qa['id'], qa['question']
				sample = {'uid': uid, 'context': context, 'question': question}
				rows.append(sample)
	return rows

def postag_func(toks, vocab):
	return [vocab[w.tag_] for w in toks if len(w.text) > 0]

def nertag_func(toks, vocab):
	return [vocab['{}_{}'.format(w.ent_type_, w.ent_iob_)] for w in toks if len(w.text) > 0]
	#return [vocab[w.ent_type_] for w in toks if len(w.text) > 0]

def tok_func(toks, vocab):
	return [vocab[w.text] for w in toks if len(w.text) > 0]

def toktext_func(toks):
	return [w.text for w in toks if len(w.text) > 0]

def match_func(question, context):
	counter = Counter(w.text.lower() for w in context)
	total = sum(counter.values())
	freq = [counter[w.text.lower()] / total for w in context]
	question_word = {w.text for w in question}
	question_lower = {w.text.lower() for w in question}
	question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
	match_origin = [1 if w in question_word else 0 for w in context]
	match_lower = [1 if w.text.lower() in question_lower else 0 for w in context]
	match_lemma = [1 if (w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma else 0 for w in context]
	features = np.asarray([freq, match_origin, match_lower, match_lemma], dtype=np.float32).T.tolist()
	return features

def build_span(context, context_token):
	p_str = 0
	p_token = 0
	t_start, t_end, t_span = -1, -1, []
	while p_str < len(context):
		if re.match('\s', context[p_str]):
			p_str += 1
			continue
		token = context_token[p_token]
		token_len = len(token)
		if context[p_str:p_str + token_len] != token:
			logger.info('dropped:',context[p_str:p_str + token_len], '/ and / ', token)
			return (-1, -1, [])
		t_span.append((p_str, p_str + token_len))
		p_str += token_len
		p_token += 1
	return (t_start, t_end, t_span)

def feature_func(sample, doc_tokend, query_tokend, vocab, vocab_tag, vocab_ner):
	# features
	fea_dict = {}
	fea_dict['uid'] = sample['uid']
	fea_dict['query_tok'] = tok_func(query_tokend, vocab)
	fea_dict['query_pos'] = postag_func(query_tokend, vocab_tag)
	fea_dict['query_ner'] = nertag_func(query_tokend, vocab_ner)
	fea_dict['doc_tok'] = tok_func(doc_tokend, vocab)
	fea_dict['doc_pos'] = postag_func(doc_tokend, vocab_tag)
	fea_dict['doc_ner'] = nertag_func(doc_tokend, vocab_ner)
	fea_dict['doc_fea'] = '{}'.format(match_func(query_tokend, doc_tokend))
	fea_dict['query_fea'] = '{}'.format(match_func(doc_tokend, query_tokend))
	doc_toks = [t.text for t in doc_tokend if len(t.text) > 0]
	query_toks = [t.text for t in query_tokend if len(t.text) > 0]
	fea_dict['doc_ctok'] = doc_toks
	fea_dict['query_ctok'] = query_toks

	_, _, span = build_span(sample['context'], doc_toks)
	fea_dict['context'] = sample['context']
	fea_dict['span'] = span
	return fea_dict

def build_data(data, vocab, vocab_tag, vocab_ner, n_threads=16):
	dropped_sample = 0
	all_data = []
	context = [reform_text(sample['context']) for sample in data]
	context_parsed = [doc for doc in NLP.pipe(context, batch_size=10000, n_threads=n_threads)]

	query = [reform_text(sample['question']) for sample in data]
	query_parsed = [question for question in NLP.pipe(query, batch_size=10000, n_threads=n_threads)]
	logger.info('Done with tokenizing')

	for sample, doc_tokend, query_tokend in tqdm.tqdm(zip(data, context_parsed, query_parsed), total=len(data)):
		fd = feature_func(sample, doc_tokend, query_tokend, vocab, vocab_tag, vocab_ner)
		if fd is None:
			dropped_sample += 1
			continue
		all_data.append(fd)
	logger.info('Got {} data sample in total {}'.format(len(all_data), len(data)))
	return all_data



def load_squad_v2(path):
	with open(path, 'r', encoding='utf-8') as f:
		dataset_json = json.load(f)
		dataset = dataset_json['data']
		return dataset



def evaluate_squad_v2(model, data):
	data.reset()
	predictions = {}
	score_list = {}
	for batch in data:
		phrase, _, scores = model.predict(batch)
		uids = batch['uids']
		for uid, pred, score in zip(uids, phrase, scores):
			predictions[uid] = pred
			score_list[uid] = score
	return predictions, score_list



def build_vocab(test_data, tr_vocab, n_threads=16):
	nlp = spacy.load('en', disable=['vectors', 'textcat', 'parser', 'tagger', 'ner'])
	text = [reform_text(sample['context']) for sample in test_data] + [reform_text(sample['question']) for sample in test_data]
	parsed = [doc for doc in nlp.pipe(text, batch_size=10000, n_threads=n_threads)]
	tokens = [w.text for doc in parsed for w in doc if len(w.text) > 0]
	new_vocab = list(set([w for w in tokens if w not in tr_vocab and w in glove_vocab]))
	for w in new_vocab:
		tr_vocab.add(w)
	return tr_vocab


if __name__ == '__main__':

	with open('data_v2/squad_meta_v2_train_only.pick', 'rb') as f:
		train_only_meta = pickle.load(f)

	tr_vocab = train_only_meta['vocab']
	vocab_tag = train_only_meta['vocab_tag']
	vocab_ner = train_only_meta['vocab_ner']
	logger.info('loaded meta data')

	glove_vocab = load_glove_vocab(glove_path, glove_dim)
	logger.info('loaded glove vector')

	# setting up spacy
	NLP = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

	# load test data
	test_data = load_data(test_file)

	# training vocab - does it contain embedding?
	tr_vocab = Vocabulary.build(tr_vocab.get_vocab_list())

	test_vocab = build_vocab(test_data, tr_vocab, n_threads=n_threads)
	
	logger.info('Collected vocab')

	test_embedding = build_embedding(glv, test_vocab, glove_dim)
	logger.info('Got embedding')
	test_data = build_data(test_data, test_vocab, vocab_tag, vocab_ner, n_threads=n_threads)

	dev_data = BatchGen(test_data, batch_size, have_gpu, is_train=False, with_label=True)
	#batches.reset()
	#batches = list(batches)


	model_path = model_root+'v2_FGSM_max10_original_200_25.pt'

	checkpoint = torch.load(model_path)


	opt = checkpoint['config']
	set_environment(opt['seed'], have_gpu)
	opt['covec_path'] = mtlstm_path
	opt['cuda'] = have_gpu
	opt['multi_gpu'] = False
	opt['max_len'] = max_len
	state_dict = checkpoint['state_dict']
	model = DocReaderModel(opt, state_dict = state_dict)
	model.setup_eval_embed(torch.Tensor(test_embedding))
	logger.info('Loaded model!')

	if have_gpu:
		model.cuda()

	results, score_list = evaluate_squad_v2(model, dev_data)

	dev_gold = load_squad_v2(test_file)
	
	results = my_evaluation(dev_gold, results, score_list, 0.4)
	logger.info('{}'.format(results))