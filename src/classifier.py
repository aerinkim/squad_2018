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
        #  Classifier(query_mem_hidden_size, opt['label_size'], opt=opt, prefix='classifier', dropout=my_dropout)
        super(Classifier, self).__init__()
        self.opt = opt
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(prefix), 0))
        else:
            self.dropout = dropout
        self.merge_opt = opt.get('{}_merge_opt'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)

        
        if self.merge_opt == 1:
            self.proj = nn.Linear(x_size * 4, 1000)
            self.proj15 = nn.Linear(1000, 1000)
            self.proj2 = nn.Linear(1000, y_size)
        else:
            self.proj = nn.Linear(x_size * 2, 1000)
            self.proj15 = nn.Linear(1000, 1000)
            self.proj2 = nn.Linear(1000, y_size)

        if self.weight_norm_on:
            self.proj2 = weight_norm(self.proj2)

        self.relu = nn.ReLU()


    def forward(self, x1, x2, mask=None):
        # self.classifier(doc_sum, query_mem, doc_mask)
        if self.merge_opt == 1:
            x = torch.cat([x1, x2, (x1 - x2).abs(), x1 * x2], 1)
        else:
            x = torch.cat([x1, x2], 1)

        x = self.dropout(x)
        scores = self.proj(x)
        scores = self.dropout(scores)
        scores = self.relu(scores)
        scores = self.proj15(scores)
        scores = self.dropout(scores)
        scores = self.relu(scores)
        scores = self.proj2(scores)

        return scores
