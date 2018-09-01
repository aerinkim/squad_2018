import re
import os
import json
import spacy
import numpy as np
import logging
import random
import tqdm
import pickle
from my_utils.tokenizer import Vocabulary, reform_text, build_vocab, END
from my_utils.word2vec_utils import load_glove_vocab, build_embedding
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from my_utils.process_data_utils import *
from config_v2 import set_args

"""
This script is to preproces SQuAD dataset.
"""

def load_data(path, is_train=True):
    rows = []
    with open(path, encoding="utf8") as f:
        data = json.load(f)['data']
    cnt = 0
    for article in tqdm.tqdm(data, total=len(data)):
        for paragraph in article['paragraphs']:
            cnt += 1
            context = paragraph['context']
            context = '{} {}'.format(context, END)
            for qa in paragraph['qas']:
                uid, question = str(qa['id']), qa['question']
                is_impossible = qa.get('is_impossible', False)
                label = 1 if is_impossible else 0
                answers = qa.get('answers', [])
                if is_train:
                    if label < 1 and len(answers) < 1:
                        continue
                    if len(answers) > 0:
                        answer = answers[0]['text'].rstrip()
                        answer_start = int(answers[0]['answer_start'])
                        answer_end = answer_start + len(answer)
                        sample = {'uid': uid, 'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'answer_end':answer_end, 'label': label}
                    else:
                        answer = END
                        answer_start = len(context) - len(END)
                        answer_end = len(context)
                        sample = {'uid': uid, 'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'answer_end':answer_end, 'label': label}
                    rows.append(sample)
                else:
                    sample = {'uid': uid, 'context': context, 'question': question, 'answer': answers, 'answer_start': -1, 'answer_end':-1, 'label': 0}
                    rows.append(sample)
    return rows

def main():
    args = set_args()
    global logger
    logger = create_logger(__name__, to_disk=True, log_file=args.log_file)
    logger.info('~Processing SQuAD V2 dataset~')
    train_path = os.path.join(args.data_dir, 'train-v2.0.json')
    valid_path = os.path.join(args.data_dir, 'dev-v2.0.json')
    glove_path = os.path.join(args.data_dir, args.glove)
    glove_dim = args.glove_dim
    logger.info('The path of training data: {}'.format(train_path))
    logger.info('The path of validation data: {}'.format(valid_path))
    logger.info('{}-dim word vector path: {}'.format(glove_dim, glove_path))
    set_environment(args.seed)
    logger.info('Loading glove vocab.')
    glove_vocab = load_glove_vocab(glove_path, glove_dim)

    ## load data
    train_data = load_data(train_path)
    logger.info('Loaded train data: {} samples.'.format(len(train_data)))

    valid_data = load_data(valid_path, False)
    logger.info('Loaded dev data: {} samples.'.format(len(valid_data)))
    
    # build vocab
    logger.info('Build vocabulary')
    vocab, vocab_tag, vocab_ner = build_vocab(train_data, glove_vocab, sort_all=args.sort_all, thread=args.threads, clean_on=True)
    logger.info('size vocab/tag/ner are: {}, {}, {}'.format(len(vocab), len(vocab_tag), len(vocab_ner)))
    meta_path = os.path.join(args.data_dir, args.meta)
    logger.info('building embedding')
    embedding = build_embedding(glove_path, vocab, glove_dim)
    meta = {'vocab': vocab, 'vocab_tag': vocab_tag, 'vocab_ner': vocab_ner, 'embedding': embedding}

    # If you want to check vocab token IDs, etc., load the meta file below (squad_meta.pick).
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    logger.info('started the function build_data')
    dev_fout = os.path.join(args.data_dir, args.dev_data)
    build_data(valid_data, vocab, vocab_tag, vocab_ner, dev_fout, False, thread=args.threads)
    logger.info('build_data: {} completed.'.format(dev_fout))
    train_fout = os.path.join(args.data_dir, args.train_data)
    build_data(train_data, vocab, vocab_tag, vocab_ner, train_fout, True, thread=args.threads)
    logger.info('build_data: {} completed.'.format(train_fout))

if __name__ == '__main__':
    main()
