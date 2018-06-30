#/usr/bin/env python3
import argparse
import multiprocessing
import torch
"""
Configuration file
"""
def model_config(parser):
    parser.add_argument('--vocab_size', type=int, default=0)
    parser.add_argument('--wemb_dim', type=int, default=300)
    parser.add_argument('--covec_on', action='store_false')
    parser.add_argument('--embedding_dim', type=int, default=300)

    # pos
    parser.add_argument('--no_pos', dest='pos_on', action='store_false')
    parser.add_argument('--pos_vocab_size', type=int, default=56)
    parser.add_argument('--pos_dim', type=int, default=12)
    parser.add_argument('--no_ner', dest='ner_on', action='store_false')
    parser.add_argument('--ner_vocab_size', type=int, default=19)
    parser.add_argument('--ner_dim', type=int, default=8)
    parser.add_argument('--no_feat', dest='feat_on', action='store_false')
    parser.add_argument('--num_features', type=int, default=4)
    # q->p
    parser.add_argument('--prealign_on', action='store_false')
    parser.add_argument('--prealign_head', type=int, default=1)
    parser.add_argument('--prealign_att_dropout', type=float, default=0)
    parser.add_argument('--prealign_norm_on', action='store_false')
    parser.add_argument('--prealign_proj_on', action='store_true')
    parser.add_argument('--prealign_bidi', action='store_false')
    parser.add_argument('--prealign_hidden_size', type=int, default=125)
    parser.add_argument('--prealign_share', action='store_false')
    parser.add_argument('--prealign_residual_on', action='store_true')
    parser.add_argument('--prealign_scale_on', action='store_false')
    parser.add_argument('--prealign_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--prealign_activation', type=str, default='relu')

    parser.add_argument('--pwnn_on', action='store_false')
    parser.add_argument('--pwnn_hidden_size', type=int, default=125)

    ##contextual encoding
    parser.add_argument('--contextual_hidden_size', type=int, default=125)
    parser.add_argument('--contextual_cell_type', type=str, default='lstm')
    parser.add_argument('--contextual_weight_norm_on', action='store_true')
    parser.add_argument('--contextual_maxout_on', action='store_true')
    parser.add_argument('--contextual_residual_on', action='store_true')
    parser.add_argument('--contextual_encoder_share', action='store_true')
    parser.add_argument('--contextual_num_layers', type=int, default=2)

    ## mem setting
    parser.add_argument('--msum_hidden_size', type=int, default=125)
    parser.add_argument('--msum_cell_type', type=str, default='lstm')
    parser.add_argument('--msum_weight_norm_on', action='store_false')
    parser.add_argument('--msum_maxout_on', action='store_true')
    parser.add_argument('--msum_residual_on', action='store_true')
    parser.add_argument('--msum_lexicon_input_on', action='store_true')
    parser.add_argument('--msum_num_layers', type=int, default=1)

    # attention
    parser.add_argument('--deep_att_lexicon_input_on', action='store_false')
    parser.add_argument('--deep_att_hidden_size', type=int, default=250)
    parser.add_argument('--deep_att_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--deep_att_activation', type=str, default='relu')
    parser.add_argument('--deep_att_norm_on', action='store_false')
    parser.add_argument('--deep_att_proj_on', action='store_true')
    parser.add_argument('--deep_att_residual_on', action='store_true')
    parser.add_argument('--deep_att_share', action='store_false')
    parser.add_argument('--deep_att_opt', type=int, default=0)

    # self attn
    parser.add_argument('--self_attention_on', action='store_false')
    parser.add_argument('--self_att_hidden_size', type=int, default=75)
    parser.add_argument('--self_att_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--self_att_activation', type=str, default='relu')
    parser.add_argument('--self_att_norm_on', action='store_false')
    parser.add_argument('--self_att_proj_on', action='store_true')
    parser.add_argument('--self_att_residual_on', action='store_true')
    parser.add_argument('--self_att_dropout', type=float, default=0.1)
    parser.add_argument('--self_att_drop_diagonal', action='store_false')
    parser.add_argument('--self_att_share', action='store_false')

    # query summary
    parser.add_argument('--query_sum_att_type', type=str, default='linear',
                        help='linear/mlp')
    parser.add_argument('--query_sum_norm_on', action='store_false')

    parser.add_argument('--san_on', action='store_false')
    parser.add_argument('--max_len', type=int, default=5)
    parser.add_argument('--decoder_ptr_update_on', action='store_true')
    parser.add_argument('--decoder_num_turn', type=int, default=5)
    parser.add_argument('--decoder_mem_type', type=int, default=1)
    parser.add_argument('--decoder_mem_drop_p', type=float, default=0.4)
    parser.add_argument('--decoder_opt', type=int, default=0)
    parser.add_argument('--decoder_att_hidden_size', type=int, default=125)
    parser.add_argument('--decoder_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--decoder_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--decoder_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--decoder_weight_norm_on', action='store_false')
    return parser

def data_config(parser):
    parser.add_argument('--log_file', default='san.log', help='path for log file.')
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--meta', default='squad_meta_v1.pick', help='path to preprocessed meta file.')
    parser.add_argument('--train_data', default='train_data_v1.json',
                        help='path to preprocessed training data file.')
    parser.add_argument('--dev_data', default='dev_data_v1.json',
                        help='path to preprocessed validation data file.')
    parser.add_argument('--dev_gold', default='dev-v1.1.json',
    #parser.add_argument('--dev_gold', default='data/dev-v2.0.json',
                        help='path to preprocessed validation data file.')
    parser.add_argument('--covec_path', default='MT-LSTM.pt')
    parser.add_argument('--glove', default='data/glove.840B.300d.txt',
                        help='path to word vector file.')
    parser.add_argument('--glove_dim', type=int, default=300,
                        help='word vector dimension.')
    parser.add_argument('--sort_all', action='store_true',
                        help='sort the vocabulary by frequencies of all words.'
                             'Otherwise consider question words first.')
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                        help='number of threads for preprocessing.')
    return parser

def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=50)
    parser.add_argument('--epoches', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size_eval', type=int, default=30)
    parser.add_argument('--expect_version', default='v1.0')
    parser.add_argument('--resume')
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.4)
    parser.add_argument('--dropout_emb', type=float, default=0.4)
    parser.add_argument('--dropout_w', type=float, default=0.05)
    parser.add_argument('--unk_id', type=int, default=1)
    parser.add_argument('--na_prob_thresh', '-t', type=float, default=1.0,
                        help='Predict "" if no-answer probability exceeds this (default = 1.0).')

    # scheduler
    parser.add_argument('--no_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--fix_embeddings', action='store_true', help='if true, `tune_partial` will be ignored. This will remove the gradient to the embedding layer completely.')
    parser.add_argument('--tune_partial', type=int, default=1000, help='finetune top-x embeddings (including <PAD>, <UNK>). This will remove the gradient to all embeddings but the x most frequent words.')
    parser.add_argument('--model_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2013,
                        help='random seed for data shuffling, embedding init, etc.')
    return parser

def set_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    return args
