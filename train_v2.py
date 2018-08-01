import re
import os
import sys
import random
import string
import logging
import argparse
import json
import torch
import msgpack
import numpy as np
from shutil import copyfile
from datetime import datetime
from collections import Counter, defaultdict
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGen
from config_v2 import set_args
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from my_utils.squad_eval_v2 import my_evaluation
from my_utils.data_utils import load_squad_v2, evaluate_squad_v2

args = set_args()

# set model dir
model_dir = args.model_dir
data_dir = args.data_dir

if args.philly_on:
    model_dir = os.path.join(args.modelDir, '../checkpoint')
    os.makedirs(model_dir, exist_ok=True)
    model_dir = os.path.abspath(model_dir)
    data_dir = args.dataDir
else:
    os.makedirs(model_dir, exist_ok=True)
    model_dir = os.path.abspath(model_dir)

# set environment
set_environment(args.seed, args.cuda)
# setup logger
log_path = args.log_file
if args.philly_on:
    log_path = os.path.join(args.modelDir, 'san.log')
logger =  create_logger(__name__, to_disk=True, log_file=log_path)

def main():
    logger.info('Launching the SAN')
    opt = vars(args)
    # update data dir
    opt['data_dir'] = data_dir
    logger.info('Loading data')
    embedding, opt = load_meta(opt, os.path.join(data_dir, args.meta))
    batch_size = args.batch_size
    if args.elmo_on:
        batch_size = int(batch_size/2)

    train_data = BatchGen(os.path.join(data_dir, args.train_data),
                          batch_size=batch_size,
                          gpu=args.cuda,
                          with_label=True,
                          elmo_on=args.elmo_on)
    dev_data = BatchGen(os.path.join(data_dir, args.dev_data),
                          batch_size=args.batch_size_eval,
                          gpu=args.cuda, is_train=False, with_label=True,
                          elmo_on=args.elmo_on)
    logger.info('#' * 20)
    logger.info(opt)
    logger.info('#' * 20)

    model = DocReaderModel(opt, embedding)
    # model meta str
    headline = '############# Model Arch of SAN #############'
    # print network
    logger.info('\n{}\n{}\n'.format(headline, model.network))
    model.setup_eval_embed(embedding)

    logger.info("Total number of params: {}".format(model.total_param))
    if args.cuda:
        model.cuda()

    best_em_score, best_f1_score = 0.0, 0.0
    dev_gold = load_gold(os.path.join(data_dir, args.dev_gold))
    print("PROGRESS: 00.00%")
    for epoch in range(0, args.epoches):
        logger.warning('At epoch {}'.format(epoch))
        train_data.reset()
        start = datetime.now()
        for i, batch in enumerate(train_data):
            model.update(batch)
            if (model.updates) % args.log_per_updates == 0 or model.updates == 1:
                logger.info('updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(train_data) - i - 1)).split('.')[0]))

        # dev eval
        results, score_list = evaluate_squad_v2(model, dev_data)
        output_path = os.path.join(model_dir, 'dev_output_{}.json'.format(epoch))
        with open(output_path, 'w') as f:
            json.dump(results, f)

        output_path = os.path.join(model_dir, 'dev_output_no_prob_{}.json'.format(epoch))
        with open(output_path, 'w') as f:
            json.dump(score_list, f)

        results = my_evaluation(dev_gold, results, score_list, args.na_prob_thresh)
        logger.info('{}'.format(results))
        em, f1 = results['exact'], results['f1']

        # setting up scheduler
        if model.scheduler is not None:
            logger.info('scheduler_type {}'.format(opt['scheduler_type']))
            if opt['scheduler_type'] == 'rop':
                model.scheduler.step(f1, epoch=epoch)
            else:
                model.scheduler.step()
        # save
        model_file = os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
        if not args.philly_on:
            model.save(model_file)

        if em + f1 > best_em_score + best_f1_score:
            for i in range(0, 10):
                try:
                    # save on philly
                    model.save(os.path.join(args.modelDir, 'best_checkpoint.pt'))
                    logger.info('Saved the new best model and prediction')
                    break
                except:
                    continue
            best_em_score, best_f1_score = em, f1
        logger.warning("Epoch {0} - dev EM: {1:.3f} F1: {2:.3f} (best EM: {3:.3f} F1: {4:.3f})".format(epoch, em, f1, best_em_score, best_f1_score))
        print("PROGRESS: {0:.2f}%".format(100.0 * (epoch + 1) / args.epoches - 1.0))

if __name__ == '__main__':
    main()
