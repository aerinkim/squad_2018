import re
import warnings
import tqdm
import logging
import unicodedata
from collections import Counter
from functools import partial
from multiprocessing import Pool as ThreadPool


logger = logging.getLogger(__name__)

DUMMY = 'DUMMMMMY'
PAD = '<PAD>'
UNK = '<UNK>'
STA= '<BOS>'
END = '<EOS>'

PAD_ID = 0
UNK_ID = 1
STA_ID = 2
END_ID = 3

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '

def reform_text(text):
    text = re.sub(u'-|¢|¥|€|£|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

class Vocabulary(object):
    INIT_LEN = 4
    def __init__(self, neat=False):
        self.neat = neat
        if not neat:
            self.tok2ind = {PAD: PAD_ID, UNK: UNK_ID, STA: STA_ID, END: END_ID}
            self.ind2tok = {PAD_ID: PAD, UNK_ID: UNK, STA_ID: STA, END_ID:END}
        else:
            self.tok2ind = {}
            self.ind2tok = {}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return key in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, -1) if self.neat else self.ind2tok.get(key, UNK)
        if type(key) == str:
            return self.tok2ind.get(key, None) if self.neat else self.tok2ind.get(key,self.tok2ind.get(UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def get_vocab_list(self, with_order=True):
        if with_order:
            words = [self[k] for k in range(0, len(self))]
        else:
            words = [k for k in self.tok2ind.keys()
                      if k not in {PAD, UNK, STA, END}]
        return words

    def toidx(self, tokens):
        return [self[tok] for tok in tokens]

    def copy(self):
        """Deep copy
        """
        new_vocab = Vocabulary(self.neat)
        for w in self:
            new_vocab.add(w)
        return new_vocab

    def build(words, neat=False):
        vocab = Vocabulary(neat)
        for w in words: vocab.add(w)
        return vocab

def build_vocab(data, glove_vocab=None, sort_all=False, thread=24, clean_on=False):
    nlp = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

    print('Collect vocab/pos counter/ner counter')
    # docs
    docs = [reform_text(sample['context']) for sample in data]
    doc_tokened = [doc for doc in nlp.pipe(docs, batch_size=10000, n_threads=thread)]
    print('Done with doc tokenize')
    questions = [reform_text(sample['question']) for sample in data]
    questions_tokened = [question for question in nlp.pipe(questions, batch_size=10000, n_threads=thread)]
    print('Done with question tokenize')

    tag_counter = Counter()
    ner_counter = Counter()
    if sort_all:
        counter = Counter()
        merged = doc_tokened + questions_tokened
        for tokened in tqdm.tqdm(merged, total=len(data)):
            counter.update([w.text for w in tokened if len(w.text) > 0])
            tag_counter.update([w.tag_ for w in tokened if len(w.text) > 0])
            ner_counter.update(['{}_{}'.format(w.ent_type_, w.ent_iob_) for w in tokened])
        vocab = sorted([w for w in counter if w in glove_vocab], key=counter.get, reverse=True)
    else:
        query_counter = Counter()
        doc_counter = Counter()

        for tokened in tqdm.tqdm(doc_tokened, total=len(doc_tokened)):
            doc_counter.update([w.text for w in tokened if len(w.text) > 0])
            tag_counter.update([w.tag_ for w in tokened if len(w.text) > 0])
            ner_counter.update(['{}_{}'.format(w.ent_type_, w.ent_iob_) for w in tokened])

        for tokened in tqdm.tqdm(questions_tokened, total=len(questions_tokened)):
            query_counter.update([w.text for w in tokened if len(w.text) > 0])
            tag_counter.update([w.tag_ for w in tokened if len(w.text) > 0])
            ner_counter.update(['{}_{}'.format(w.ent_type_, w.ent_iob_) for w in tokened])
        counter = query_counter + doc_counter
        # sort query words
        vocab = sorted([w for w in query_counter if w in glove_vocab], key=query_counter.get, reverse=True)
        vocab += sorted([w for w in doc_counter.keys() - query_counter.keys() if w in glove_vocab], key=counter.get, reverse=True)
    tag_counter = sorted([w for w in tag_counter], key=tag_counter.get, reverse=True)
    ner_counter = sorted([w for w in ner_counter], key=ner_counter.get, reverse=True)

    total = sum(counter.values())
    matched = sum(counter[w] for w in vocab)
    print('Raw vocab size vs vocab in glove: {0}/{1}'.format(len(counter), len(vocab)))
    print('OOV rate:{0:.4f}={1}/{2}'.format(100.0 * (total - matched)/total, (total - matched), total))
    vocab = Vocabulary.build(vocab)
    tag_vocab = Vocabulary.build(tag_counter)
    ner_vocab = Vocabulary.build(ner_counter)
    print('final vocab size: {}'.format(len(vocab)))
    print('POS Tag vocab size: {}'.format(len(tag_vocab)))
    print('NER Tag vocab size: {}'.format(len(ner_vocab)))

    return vocab, tag_vocab, ner_vocab