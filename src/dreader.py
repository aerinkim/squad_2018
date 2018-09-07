import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .recurrent import OneLayerBRNN, ContextualEmbed
from .dropout_wrapper import DropoutWrapper
from .encoder import LexiconEncoder
from .similarity import DeepAttentionWrapper, FlatSimilarityWrapper, SelfAttnWrapper
from .similarity import AttentionWrapper
from .san import SAN
from .classifier import Classifier

class DNetwork(nn.Module):
    """Network for SAN doc reader."""
    def __init__(self, opt, embedding=None, padding_idx=0):
        super(DNetwork, self).__init__()
        my_dropout = DropoutWrapper(opt['dropout_p'], opt['vb_dropout'])
        self.dropout = my_dropout

        self.lexicon_encoder = LexiconEncoder(opt, embedding=embedding, dropout=my_dropout)
        query_input_size = self.lexicon_encoder.query_input_size
        doc_input_size = self.lexicon_encoder.doc_input_size
        elmo_size = self.lexicon_encoder.elmo_size
        print('Lexicon encoding size for query and doc are:{}', doc_input_size, query_input_size)
        covec_size = self.lexicon_encoder.covec_size
        embedding_size = self.lexicon_encoder.embedding_dim

        # Contextual Encoders
        contextual_share = opt.get('contextual_encoder_share', False)
        context_num_layer = opt['contextual_num_layers']
        prefix = 'contextual'
        self.doc_encoders = nn.ModuleList()
        self.query_encoders = nn.ModuleList()

        assert doc_input_size == query_input_size
        tmp_input_size = doc_input_size
        hidden_size = 0
        for i in range(0, context_num_layer):
            input_size = tmp_input_size + covec_size if i < 2 else tmp_input_size
            input_size += elmo_size
            doc_encoder = OneLayerBRNN(input_size, opt['contextual_hidden_size'], prefix=prefix, opt=opt, dropout=my_dropout)
            if contextual_share:
                query_encoder = doc_encoder
            else:
                query_encoder = OneLayerBRNN(input_size, opt['contextual_hidden_size'], prefix=prefix, opt=opt, dropout=my_dropout)
            self.doc_encoders.append(doc_encoder)
            self.query_encoders.append(query_encoder)
            assert doc_encoder.output_size == query_encoder.output_size
            hidden_size += doc_encoder.output_size
            tmp_input_size = doc_encoder.output_size

        doc_hidden_size, query_hidden_size = hidden_size, hidden_size

        self.query_understand = OneLayerBRNN(query_hidden_size, opt['msum_hidden_size'], prefix='msum', opt=opt, dropout=my_dropout)
        doc_attn_size = doc_hidden_size + covec_size + embedding_size
        query_attn_size = query_hidden_size + covec_size + embedding_size
        num_layers = context_num_layer + 1
        if opt['elmo_att_on']:
            doc_attn_size += elmo_size
            query_attn_size += elmo_size

        prefix = 'deep_att'
        self.deep_attn = DeepAttentionWrapper(doc_attn_size, query_attn_size, num_layers, prefix, opt, my_dropout)

        doc_und_size = doc_hidden_size + query_hidden_size + self.query_understand.output_size
        self.doc_understand = OneLayerBRNN(doc_und_size, opt['msum_hidden_size'], prefix='msum', opt=opt, dropout=my_dropout)
        query_mem_hidden_size = self.query_understand.output_size
        doc_mem_hidden_size = self.doc_understand.output_size

        if opt['self_attention_on']:
            att_size = embedding_size + covec_size + doc_hidden_size + query_hidden_size + self.query_understand.output_size + self.doc_understand.output_size
            if opt['elmo_self_att_on']: att_size += elmo_size
            self.doc_self_attn = AttentionWrapper(att_size, att_size, prefix='self_att', opt=opt, dropout=my_dropout)
            doc_mem_hidden_size = doc_mem_hidden_size * 2
            self.doc_mem_gen = OneLayerBRNN(doc_mem_hidden_size, opt['msum_hidden_size'], 'msum', opt, my_dropout)
            doc_mem_hidden_size = self.doc_mem_gen.output_size
        # Question merging
        self.query_sum_attn = SelfAttnWrapper(query_mem_hidden_size, prefix='query_sum', opt=opt, dropout=my_dropout)
        self.decoder = SAN(doc_mem_hidden_size, query_mem_hidden_size, opt, prefix='decoder', dropout=my_dropout)
        if opt.get('extra_loss_on', False):
            self.doc_sum_attn = SelfAttnWrapper(doc_mem_hidden_size, prefix='query_sum', opt=opt, dropout=my_dropout)
            self.classifier = Classifier(query_mem_hidden_size, opt['label_size'], opt=opt, prefix='classifier', dropout=my_dropout)
        else:
            self.classifier = None
        self.opt = opt

    def forward(self, batch):
        doc_input, query_input,\
        doc_emb, query_emb,\
        doc_cove_low, doc_cove_high,\
        query_cove_low, query_cove_high,\
        doc_mask, query_mask,\
        doc_elmo, query_elmo = self.lexicon_encoder(batch)

        query_list, doc_list = [], []
        query_list.append(query_input)
        doc_list.append(doc_input)

        # doc encode
        doc_represenations = []
        for i, doc_encoder in enumerate(self.doc_encoders):
            if i == 0:
                doc_cont_input = torch.cat([doc_input, doc_cove_low], 2)
            elif i == 1:
                doc_cont_input = torch.cat([doc_input, doc_cove_high], 2)
            else:
                doc_cont_input = doc_input
            if self.lexicon_encoder.elmo_on:
                doc_cont_input = torch.cat([doc_cont_input, doc_elmo[i]], 2)
            doc_rep = doc_encoder(doc_cont_input, doc_mask)
            doc_rep = self.dropout(doc_rep)
            doc_input = doc_rep
            doc_represenations.append(doc_rep)

        query_represenations = []
        for i, query_encoder in enumerate(self.query_encoders):
            if i == 0:
                query_cont_iput = torch.cat([query_input, query_cove_low], 2)
            elif i == 1:
                query_cont_iput = torch.cat([query_input, query_cove_high], 2)
            else:
                query_cont_iput = query_input
            if self.lexicon_encoder.elmo_on:
                query_cont_iput = torch.cat([query_cont_iput, query_elmo[i]], 2)

            query_rep = query_encoder(query_cont_iput, query_mask)
            query_rep = self.dropout(query_rep)
            query_input = query_rep
            query_represenations.append(query_rep)

        query_mem_hiddens = self.query_understand(torch.cat(query_represenations, 2), query_mask)
        query_mem_hiddens = self.dropout(query_mem_hiddens)
        query_list = query_represenations + [query_mem_hiddens]
        doc_list = doc_represenations
        if self.opt['elmo_on'] and self.opt['elmo_att_on']:
            idx = -2 if self.opt['elmo_self_att_on'] else -1
            query_att_input = torch.cat([query_emb, query_cove_high, query_elmo[idx]] + query_represenations, 2)
            doc_att_input = torch.cat([doc_emb, doc_cove_high, doc_elmo[idx]] + doc_list, 2)
        else:
            query_att_input = torch.cat([query_emb, query_cove_high] + query_represenations, 2)
            doc_att_input = torch.cat([doc_emb, doc_cove_high] + doc_list, 2)

        doc_attn_hiddens = self.deep_attn(doc_att_input, query_att_input, query_list, query_mask)
        doc_attn_hiddens = self.dropout(doc_attn_hiddens)

        doc_mem_hiddens = self.doc_understand(torch.cat([doc_attn_hiddens] + doc_list, 2), doc_mask)
        doc_mem_hiddens = self.dropout(doc_mem_hiddens)
        doc_mem_inputs = torch.cat([doc_attn_hiddens] + doc_list, 2)

        if self.opt['self_attention_on']:
            if self.opt['elmo_on'] and self.opt['elmo_self_att_on']:
                doc_att = torch.cat([doc_mem_inputs, doc_mem_hiddens, doc_cove_high, doc_emb, doc_elmo[-1]], 2)
            else:
                doc_att = torch.cat([doc_mem_inputs, doc_mem_hiddens, doc_cove_high, doc_emb], 2)
            doc_self_hiddens = self.doc_self_attn(doc_att, doc_att, doc_mask, x3=doc_mem_hiddens)
            doc_mem = self.doc_mem_gen(torch.cat([doc_mem_hiddens, doc_self_hiddens], 2), doc_mask)
        else:
            doc_mem = doc_mem_hiddens
        query_mem = self.query_sum_attn(query_mem_hiddens, query_mask)
        doc_mem = self.dropout(doc_mem)
        start_scores, end_scores = self.decoder(doc_mem, query_mem, doc_mask)
        pred_score = None
        if self.classifier is not None:
            doc_sum = self.doc_sum_attn(doc_mem, doc_mask)
            #import pdb; pdb.set_trace()
            pred_score = F.sigmoid(self.classifier(doc_sum, query_mem, doc_mask))
        return start_scores, end_scores, pred_score



