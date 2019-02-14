import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import math
from collections import defaultdict
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
from my_utils.utils import AverageMeter
from .dreader import DNetwork
#from .adversarial_loss import *

logger = logging.getLogger(__name__)

class DocReaderModel(object):
    def __init__(self, opt, embedding=None, state_dict=None):
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict and 'updates' in state_dict else 0
        self.eval_embed_transfer = True
        self.train_loss = AverageMeter()
        self.embedding = embedding
        self.network = DNetwork(opt, embedding)
        #self.adversarial_loss = adversarial_loss

        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            for k, v in list(self.network.state_dict().items()):
                if k not in state_dict['network']:
                    state_dict['network'][k] = v
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]

        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          opt['learning_rate'],
                                          weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters,
                                        opt['learning_rate'],
                                        weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adadelta':
            self.optimizer = optim.Adadelta(parameters,
                                            opt['learning_rate'],
                                            rho=0.95)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        if opt['fix_embeddings']:
            wvec_size = 0
        else:
            wvec_size = (opt['vocab_size'] - opt['tune_partial']) * opt['embedding_dim']
        
        if opt.get('have_lr_scheduler', False):
            if opt.get('scheduler_type', 'rop') == 'rop':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=opt['lr_gamma'], patience=3)
            elif opt.get('scheduler_type', 'rop') == 'exp':
                self.scheduler = ExponentialLR(self.optimizer, gamma=opt.get('lr_gamma', 0.5))
            else:
                milestones = [int(step) for step in opt.get('multi_step_lr', '10,20,30').split(',')]
                self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=opt.get('lr_gamma'))
        else:
            self.scheduler = None
        self.total_param = sum([p.nelement() for p in parameters]) - wvec_size


    def adversarial_loss(self, batch, loss, y, no_answer):
        """
        Input: 
                - batch: batch that is shared with update method. In this code, batch contains not only embedding but also other things such as POS tag, NER etc.
                - loss : loss function that we will propagate the shit. 
                - y : answer span y label
                - no_answer : the label that tells you whether or not the question has the answer
        Output:
                - adv_loss scalar tensor.
        """
        self.optimizer.zero_grad() # Remove accumulated grads from the previous batch.
        loss.backward(retain_graph=True) # backprop. Every derivative of realted variables of loss is caluclated.

        grad = self.network.lexicon_encoder.embedding.weight.grad.data.clone() # d(loss)/d((embedding) # You need to call embedding.weight. #shape:[vocab #, word emb dimension]
        #grad.detach_() # This stops optimizer to optimize embdding. 

        """
        #Let's sanity check if it's only contains 32 rows.
        #print((grad!=0).sum(dim=0)) #[1,300] tensor that # of non zero elems in the N

        #Let's check the token and grad matching.
        print(np.unique(torch.nonzero((grad!=0))[:,0]))
        a = batch['doc_tok'].cpu().numpy().flatten().tolist()
        b = batch['query_tok'].cpu().numpy().flatten().tolist()
        c = list(set(a+b))
        c.sort()
        print(c)
        """
        
        #TODO: numerically stable L2.
        perturb = F.normalize(grad, p=2, dim =1) * 100.0 # hyper params will be 0.5 0.6 0.7 1
        #import pdb; pdb.set_trace()
        adv_embedding = self.network.lexicon_encoder.embedding.weight + perturb
        # cache the original embedding's weight (to revert it back to the original state.)
        original_emb = self.network.lexicon_encoder.embedding.weight.data
        self.network.lexicon_encoder.embedding.weight.data = adv_embedding
        # forward propagate. You are not evaluating here .You are training weights now using the perturbed input.
        adv_start, adv_end, adv_pred = self.network(batch)
        #revert it back to the original embedding. so that it doesn't get bigger for the words that appear a lot i.e. "the"
        self.network.lexicon_encoder.embedding.weight.data = original_emb 
        #Switch off the embedding training. 
        self.network.lexicon_encoder.embedding.training = False
        return F.cross_entropy(adv_start, y[0]) + F.cross_entropy(adv_end, y[1]) + F.binary_cross_entropy(adv_pred, no_answer) * self.opt.get('classifier_gamma', 1) 


    def update(self, batch):
        """
        The SAN learning algorithm is to learn a function f(Q,P) -> A, at a word level. 
        The training data is a set of the query, passage and the answer tuples <Q,P,A>.
        """
        self.network.train()
        if self.opt['cuda']:
            y = Variable(batch['start'].cuda(async=True), requires_grad=False), Variable(batch['end'].cuda(async=True), requires_grad=False)
            if self.opt.get('extra_loss_on', False):
                label = Variable(batch['label'].cuda(async=True), requires_grad=False)
        else:
            y = Variable(batch['start'], requires_grad=False), Variable(batch['end'], requires_grad=False)
            if self.opt.get('extra_loss_on', False):
                label = Variable(batch['label'], requires_grad=False)

        # span prediction: start- a start token of the answer span. end - an end token of the answer span. pred - binary prediction whether or not the question is answerable.
        start, end, pred = self.network(batch) # forward propagate
        
        # This is the loss of the batch. plain vanila. This is fucken scala tensor. 
        loss = F.cross_entropy(start, y[0]) + F.cross_entropy(end, y[1])
        if batch['with_label'] and self.opt.get('extra_loss_on', False):
            loss = loss + F.binary_cross_entropy(pred, label) * self.opt.get('classifier_gamma', 1)
        # Now we do adv. loss.
        loss_adv = self.adversarial_loss(batch, loss, y, label)
        # And we're gonna jointly optimize the sum. 
        loss_total = loss + loss_adv
        print("original loss : ",loss,"    adv loss :    ",loss_adv,"     diff : ",loss_adv-loss)
        self.train_loss.update(loss_total.data[0], len(start))

        self.optimizer.zero_grad() # You have to do this to remove gradients of embedding,
        loss_total.backward(retain_graph=False)        
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])
        self.optimizer.step()
        self.updates += 1
        #self.reset_embeddings()
        self.eval_embed_transfer = True

    def predict(self, batch, top_k=1):
        self.network.eval()
        self.network.drop_emb = False
        # Transfer trained embedding to evaluation embedding
        if self.eval_embed_transfer:
            self.update_eval_embed()
            self.eval_embed_transfer = False

        start, end, lab = self.network(batch)
        start = F.softmax(start)
        end = F.softmax(end)
        start = start.data.cpu()
        end = end.data.cpu()

        # lab is used for SQuAD v2
        if lab is not None:
            lab = lab.data.cpu()

        text = batch['text']
        spans = batch['span']
        predictions = []
        best_scores = []
        label_predictions = []

        max_len = self.opt['max_len'] or start.size(1)
        doc_len = start.size(1)
        pos_enc = self.position_encoding(doc_len, max_len)
        for i in range(start.size(0)):
            scores = torch.ger(start[i], end[i])
            scores = scores * pos_enc
            scores.triu_()
            scores = scores.numpy()
            best_idx = np.argpartition(scores, -top_k, axis=None)[-top_k]
            best_score = np.partition(scores, -top_k, axis=None)[-top_k]
            s_idx, e_idx = np.unravel_index(best_idx, scores.shape)
            if batch['with_label'] and self.opt.get('extra_loss_on', False):
                label_score = float(lab[i])
                s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                answer = text[i][s_offset:e_offset]
                if s_idx == len(spans[i]) - 1:
                    answer = ''
                predictions.append(answer)
                best_scores.append(best_score)
                label_predictions.append(label_score)
            else:
                s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                predictions.append(text[i][s_offset:e_offset])
                best_scores.append(best_score)
        if self.opt.get('extra_loss_on', False):
            return (predictions, best_scores, label_predictions)
        else:
            return (predictions, best_scores)

    def setup_eval_embed(self, eval_embed, padding_idx = 0):
        self.network.lexicon_encoder.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1),
                                               padding_idx = padding_idx)
        self.network.lexicon_encoder.eval_embed.weight.data = eval_embed
        for p in self.network.lexicon_encoder.eval_embed.parameters():
            p.requires_grad = False
        self.eval_embed_transfer = True  

        if self.opt['covec_on']:
            self.network.lexicon_encoder.ContextualEmbed.setup_eval_embed(eval_embed)

    def update_eval_embed(self):
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial']
            self.network.lexicon_encoder.eval_embed.weight.data[0:offset] \
                = self.network.lexicon_encoder.embedding.weight.data[0:offset]

    def reset_embeddings(self):
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial']
            if offset < self.network.lexicon_encoder.embedding.weight.data.size(0):
                self.network.lexicon_encoder.embedding.weight.data[offset:] \
                    = self.network.lexicon_encoder.fixed_embedding

    def save(self, filename):
        # strip cove
        network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items() if k[0:4] != 'CoVe'])
        if 'eval_embed.weight' in network_state:
            del network_state['eval_embed.weight']
        if 'fixed_embedding' in network_state:
            del network_state['fixed_embedding']
        params = {
            'state_dict': {'network': network_state},
            'config': self.opt,
        }
        torch.save(params, filename)
        logger.info('model saved to {}'.format(filename))

    def cuda(self):
        self.network.cuda()

    def position_encoding(self, m, threshold=4):
        encoding = np.ones((m, m), dtype=np.float32)
        for i in range(m):
            for j in range(i, m):
                if j - i > threshold:
                    encoding[i][j] = float(1.0 / math.log(j - i + 1))
        return torch.from_numpy(encoding)
