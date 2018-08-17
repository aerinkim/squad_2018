import os
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .recurrent import BRNNEncoder, ContextualEmbed
from .dropout_wrapper import DropoutWrapper
from .common import activation
from .similarity import AttentionWrapper
from .sub_layers import PositionwiseNN
from allennlp.modules.elmo import Elmo

class LexiconEncoder(nn.Module):
    def create_embed(self, vocab_size, embed_dim, embedding_max_norm=None, padding_idx=0):
        return nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx, max_norm = embedding_max_norm)

    def create_word_embed(self, embedding=None, opt={}, prefix='wemb'):
        vocab_size = opt.get('vocab_size', 1)
        embed_dim = opt.get('{}_dim'.format(prefix), 300)
        embedding_max_norm = opt.get('embedding_max_norm', None)
        self.embedding = self.create_embed(vocab_size, embed_dim, embedding_max_norm)
        if embedding is not None:
            self.embedding.weight.data = embedding
            if opt['fix_embeddings'] or opt['tune_partial'] == 0:
                opt['fix_embeddings'] = True
                opt['tune_partial'] = 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            else:
                assert opt['tune_partial'] <= embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial']:]
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        return embed_dim

    def create_pos_embed(self, opt={}, prefix='pos'):
        vocab_size = opt.get('{}_vocab_size'.format(prefix), 56)
        embed_dim = opt.get('{}_dim'.format(prefix), 12)
        self.pos_embedding = self.create_embed(vocab_size, embed_dim)
        return embed_dim

    def create_ner_embed(self, opt={}, prefix='ner'):
        vocab_size = opt.get('{}_vocab_size'.format(prefix), 19)
        embed_dim = opt.get('{}_dim'.format(prefix), 8)
        self.ner_embedding = self.create_embed(vocab_size, embed_dim)
        return embed_dim

    def create_cove(self, vocab_size, embedding=None, embed_dim=300, padding_idx=0, opt=None):
        self.ContextualEmbed= ContextualEmbed(os.path.join(opt['data_dir'], opt['covec_path']), opt['vocab_size'], embedding=embedding, padding_idx=padding_idx)
        return self.ContextualEmbed.output_size

    def create_prealign(self, x1_dim, x2_dim, opt={}, prefix='prealign'):
        self.prealign = AttentionWrapper(x1_dim, x2_dim, prefix, opt, self.dropout)

    def create_elmo(self, opt):
        elmo_on = opt.get('elmo_on', False)
        num_layer = opt['contextual_num_layers']
        if opt['elmo_att_on']:
            num_layer += 1

        if opt['elmo_self_att_on']:
            num_layer += 1

        size = opt['elmo_size']
        self.elmo_on = elmo_on
        if elmo_on:
            self.elmo = Elmo(os.path.join(opt['data_dir'], opt['elmo_config_path']), os.path.join(opt['data_dir'], opt['elmo_weight_path']), num_layer, dropout=opt['dropout_elmo'])
        else:
            self.elmo = None
            size = 0
        return size

    def __init__(self, opt, pwnn_on=True, embedding=None, padding_idx=0, dropout=None):
        super(LexiconEncoder, self).__init__()
        doc_input_size = 0
        que_input_size = 0
        self.dropout = DropoutWrapper(opt['dropout_p']) if dropout == None else dropout
        self.dropout_emb = DropoutWrapper(opt['dropout_emb'])
        self.dropout_cov = DropoutWrapper(opt['dropout_cov'])
        # word embedding
        embedding_dim = self.create_word_embed(embedding, opt)
        self.embedding_dim = embedding_dim
        doc_input_size += embedding_dim
        que_input_size += embedding_dim

        # pre-trained contextual vector
        covec_size = self.create_cove(opt['vocab_size'], embedding, opt=opt) if opt['covec_on'] else 0
        self.covec_size = covec_size

        prealign_size = 0
        if opt['prealign_on'] and embedding_dim > 0:
            prealign_size = embedding_dim
            self.create_prealign(embedding_dim, embedding_dim, opt)
        self.prealign_size = prealign_size
        pos_size = self.create_pos_embed(opt) if opt['pos_on'] else 0
        ner_size = self.create_ner_embed(opt) if opt['ner_on'] else 0
        feat_size = opt['num_features'] if opt['feat_on'] else 0
        elmo_size = self.create_elmo(opt)
        doc_hidden_size = embedding_dim + covec_size + prealign_size + pos_size + ner_size + feat_size
        que_hidden_size = embedding_dim + covec_size + pos_size + ner_size + feat_size
        # que_hidden_size = doc_hidden_size

        if opt['prealign_bidi']:
            que_hidden_size += prealign_size
        self.pwnn_on = pwnn_on
        self.opt = opt
        if self.pwnn_on:
            # self.pwnn = PositionwiseNN(doc_hidden_size, opt['pwnn_hidden_size'], dropout)
            # doc_input_size, que_input_size = opt['pwnn_hidden_size'], opt['pwnn_hidden_size']
            self.doc_pwnn = PositionwiseNN(doc_hidden_size, opt['pwnn_hidden_size'], dropout)
            if doc_hidden_size == que_hidden_size:
                self.que_pwnn = self.doc_pwnn
            else:
                self.que_pwnn = PositionwiseNN(que_hidden_size, opt['pwnn_hidden_size'], dropout)
            doc_input_size, que_input_size = opt['pwnn_hidden_size'], opt['pwnn_hidden_size']

        self.doc_input_size = doc_input_size
        self.query_input_size = que_input_size
        self.elmo_size = elmo_size

    def patch(self, v):
        if self.opt['cuda']:
            v = Variable(v.cuda(async=True))
        else:
            v = Variable(v)
        return v

    def forward(self, batch):
        drnn_input_list = []
        qrnn_input_list = []
        emb = self.embedding if self.training else self.eval_embed

        #embedding normalization for Adversarial Training
        #emb = F.normalize(emb, p=2, dim=0)

        doc_tok = self.patch(batch['doc_tok'])
        doc_mask = self.patch(batch['doc_mask'])
        query_tok = self.patch(batch['query_tok'])
        query_mask = self.patch(batch['query_mask'])

        doc_emb, query_emb = emb(doc_tok), emb(query_tok)
        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            doc_emb = self.dropout_emb(doc_emb)
            query_emb = self.dropout_emb(query_emb)
        drnn_input_list.append(doc_emb)
        qrnn_input_list.append(query_emb)

        doc_cove_low, doc_cove_high = None, None
        query_cove_low, query_cove_high = None, None
        if self.opt['covec_on']:
            doc_cove_low, doc_cove_high = self.ContextualEmbed(doc_tok, doc_mask)
            query_cove_low, query_cove_high = self.ContextualEmbed(query_tok, query_mask)
            doc_cove_low = self.dropout_cov(doc_cove_low)
            doc_cove_high = self.dropout_cov(doc_cove_high)
            query_cove_low = self.dropout_cov(query_cove_low)
            query_cove_high = self.dropout_cov(query_cove_high)
            drnn_input_list.append(doc_cove_low)
            qrnn_input_list.append(query_cove_low)

        if self.opt['prealign_on']:
            q2d_atten = self.prealign(doc_emb, query_emb, query_mask)
            q2d_atten = self.dropout(q2d_atten)
            drnn_input_list.append(q2d_atten)
            if self.opt['prealign_bidi']:
                d2q_atten = self.prealign(query_emb, doc_emb, doc_mask)
                d2q_atten = self.dropout(d2q_atten)
                qrnn_input_list.append(d2q_atten)

        if self.opt['pos_on']:
            doc_pos = self.patch(batch['doc_pos'])
            doc_pos_emb = self.pos_embedding(doc_pos)
            doc_pos_emb = self.dropout(doc_pos_emb)
            query_pos = self.patch(batch['query_pos'])
            query_pos_emb = self.pos_embedding(query_pos)
            query_pos_emb = self.dropout(query_pos_emb)
            drnn_input_list.append(doc_pos_emb)
            qrnn_input_list.append(query_pos_emb)

        if self.opt['ner_on']:
            doc_ner = self.patch(batch['doc_ner'])
            doc_ner_emb = self.ner_embedding(doc_ner)
            doc_ner_emb = self.dropout(doc_ner_emb)
            query_ner = self.patch(batch['query_ner'])
            query_ner_emb = self.ner_embedding(query_ner)
            query_ner_emb = self.dropout(query_ner_emb)
            drnn_input_list.append(doc_ner_emb)
            qrnn_input_list.append(query_ner_emb)

        if self.opt['feat_on']:
            doc_fea = self.patch(batch['doc_fea'])
            doc_fea = self.dropout(doc_fea)
            query_fea = self.patch(batch['query_fea'])
            query_fea = self.dropout(query_fea)
            drnn_input_list.append(doc_fea)
            qrnn_input_list.append(query_fea)

        doc_input = torch.cat(drnn_input_list, 2)
        query_input = torch.cat(qrnn_input_list, 2)
        # elmo
        if self.elmo_on:
            doc_elmo = self.elmo(self.patch(batch['doc_ctok']))['elmo_representations']
            query_elmo = self.elmo(self.patch(batch['query_ctok']))['elmo_representations']
        else:
            doc_elmo = None
            query_elmo = None

        if self.pwnn_on:
            # doc_input = self.pwnn(doc_input)
            # query_input = self.pwnn(query_input)
            # doc_input = self.dropout(doc_input)
            # query_input = self.dropout(query_input)
            doc_input = self.doc_pwnn(doc_input)
            query_input = self.que_pwnn(query_input)
            doc_input = self.dropout(doc_input)
            query_input = self.dropout(query_input)
        return doc_input, query_input, doc_emb, query_emb, doc_cove_low, doc_cove_high, query_cove_low, query_cove_high, doc_mask, query_mask, doc_elmo, query_elmo
