import torch
import random
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from .dropout_wrapper import DropoutWrapper
from .similarity import FlatSimilarityWrapper


class Classifier(nn.Module):
    def __init__(self, x_size, y_size, opt, prefix='decoder', dropout=None):
        super(Classifier, self).__init__()
        self.opt = opt
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(prefix), 0))
        else:
            self.dropout = dropout
        self.merge_opt = opt.get('{}_merge_opt'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)

        if self.merge_opt == 1:
            self.proj = nn.Linear(x_size * 4, y_size)
        else:
            self.proj = nn.Linear(x_size * 2, y_size)

        if self.weight_norm_on:
            self.proj = weight_norm(self.proj)

    def forward(self, x1, x2, mask=None):
        if self.merge_opt == 1:
            x = torch.cat([x1, x2, (x1 - x2).abs(), x1 * x2], 1)
        else:
            x = torch.cat([x1, x2], 1)
        x = self.dropout(x)
        scores = self.proj(x)
        return scores


class ClassifierPN(nn.Module):
    def __init__(self, x_size, h_size, opt={}, prefix='decoder', dropout=None):
        super(ClassifierPN, self).__init__()
        self.prefix = prefix
        self.attn = FlatSimilarityWrapper(x_size, h_size, prefix, opt, dropout)
        self.opt = opt
        self.label_size = opt['label_size']
        self.classifier = Classifier(x_size, self.label_size, opt, prefix=prefix, dropout=dropout)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, h0, x_mask):
        att_scores = self.attn(x, h0, x_mask)
        x_sum = torch.bmm(F.softmax(att_scores).unsqueeze(1), x).squeeze(1)
        x_sum = self.dropout(x_sum)
        scores = self.classifier(x_sum, h0)
        return scores
