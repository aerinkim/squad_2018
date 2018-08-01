import re
import os
import json
import spacy
import numpy as np
import logging
import random
import tqdm
import pickle
from my_utils.tokenizer import Vocabulary, reform_text, normal_query, build_vocab, END
from my_utils.word2vec_utils import load_glove_vocab, build_embedding
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from config_v2 import set_args

"""
This script is to preproces SQuAD dataset.
"""
NLP = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

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
                uid, question = qa['id'], qa['question']
                is_impossible = qa.get('is_impossible', False)
                label = 1 if is_impossible else 0
                answers = qa.get('answers', [])
                if is_train:
                    if label < 1 and len(answers) < 1:
                        continue
                    if len(answers) > 0:
                        answer = answers[0]['text']
                        answer_start = answers[0]['answer_start']
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

def build_data(data, vocab, vocab_tag, vocab_ner, fout, is_train, thread=24):
    def feature_func(sample, query_tokend, doc_tokend, is_train):
        # features
        fea_dict = {}
        fea_dict['uid'] = sample['uid']
        fea_dict['label'] = sample['label']
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
        answer_start = sample['answer_start']
        answer_end = sample['answer_end']
        answer = sample['answer']
        fea_dict['doc_ctok'] = doc_toks
        fea_dict['query_ctok'] = query_toks

        start, end, span = build_span(sample['context'], answer, doc_toks, answer_start,
                                        answer_end, is_train=is_train)
        if is_train and (start == -1 or end == -1): return None
        if not is_train:
            fea_dict['context'] = sample['context']
            fea_dict['span'] = span
        fea_dict['start'] = start
        fea_dict['end'] = end
        return fea_dict

    passages = [reform_text(sample['context']) for sample in data]
    passage_tokened = [doc for doc in NLP.pipe(passages, batch_size=64, n_threads=thread)]
    logger.info('Done with document tokenize')

    question_list = [reform_text(sample['question']) for sample in data]
    question_tokened = [question for question in NLP.pipe(question_list, batch_size=64, n_threads=thread)]
    logger.info('Done with query tokenize')

    # samples

    dropped_sample = 0
    with open(fout, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            if idx % 5000 == 0: logger.info('parse {}-th sample'.format(idx))
            feat_dict = feature_func(sample, question_tokened[idx], passage_tokened[idx], is_train)
            if feat_dict is not None:
                writer.write('{}\n'.format(json.dumps(feat_dict)))
    logger.info('dropped {} in total {}'.format(dropped_sample, len(data)))

def main():
    args = set_args()
    global logger
    logger = create_logger(__name__, to_disk=True, log_file=args.log_file)
    logger.info('~Processing SQuAD V2 dataset~')
    train_path = os.path.join(args.data_dir, 'train-v2.0.json')
    valid_path = os.path.join(args.data_dir, 'dev-v2.0.json')
    logger.info('The path of training data: {}'.format(train_path))
    logger.info('The path of validation data: {}'.format(valid_path))
    logger.info('{}-dim word vector path: {}'.format(args.glove_dim, args.glove))
    glove_path = args.glove
    glove_dim = args.glove_dim
    nlp = spacy.load('en', parser=False)
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
    vocab, vocab_tag, vocab_ner = build_vocab(train_data + valid_data, glove_vocab, sort_all=args.sort_all, clean_on=True)
    logger.info('size vocab/tag/ner are: {}, {}, {}'.format(len(vocab), len(vocab_tag), len(vocab_ner)))
    meta_path = os.path.join(args.data_dir, args.meta)
    logger.info('building embedding')
    embedding = build_embedding(glove_path, vocab, glove_dim)
    meta = {'vocab': vocab, 'vocab_tag': vocab_tag, 'vocab_ner': vocab_ner, 'embedding': embedding}

    # If you want to check vocab token IDs, etc., load the meta file below (squad_meta.pick).
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    logger.info('started the function build_data')
    train_fout = os.path.join(args.data_dir, args.train_data)
    build_data(train_data, vocab, vocab_tag, vocab_ner, train_fout, True, thread=args.threads)
    dev_fout = os.path.join(args.data_dir, args.dev_data)
    build_data(valid_data, vocab, vocab_tag, vocab_ner, dev_fout, False, thread=args.threads)

if __name__ == '__main__':
    main()

