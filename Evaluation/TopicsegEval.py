# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import shutil
from string import ascii_uppercase
from tqdm import tqdm
from Evaluation.OldROUGEEval import rouge
from ThirdParty.ROUGE import pyrouge
import sys
sys.path.append('/cluster/home/qimengnan/HMNet/Evaluation')
import accuracy
import numpy as np
from shutil import copyfile
from mpi4py import MPI
import torch
import logging
import json

def write_json_res(output_file, predictions, gts):
    data = []

    # for x_id, y_id, x_token, y_token, preds, gt in zip(x_ids, y_ids, x_tokens, y_tokens, predictions, gts):
        # x_id = tokenizers[0].decode(x_id, skip_special_tokens=False) if x_id.dim() == 1 else tokenizers[0].convert_tokens_to_string(x_token)
        # y_id = tokenizers[1].decode(y_id, skip_special_tokens=False) if y_id.dim() == 1 else tokenizers[1].convert_tokens_to_string(y_token)
    for preds, gt in zip(predictions, gts):
        data.append(
            {
                # 'x_ids': x_id,
                # 'y_ids': y_id,
                'predictions': preds,
                'gt': gt
            }
        )

    json.dump(data, output_file, indent=4, ensure_ascii=False)

logger = logging.getLogger(__name__)


class TopicsegEval():
    '''
        Wrapper class for pyrouge.
        Compute ROUGE given predictions and references for summarization evaluation.
    '''
    def __init__(self, run_dir, save_dir, opt):
        self.run_dir = run_dir
        self.save_dir = save_dir
        self.opt = opt

        # use relative path to make it work on Philly
        #self.pyrouge_dir = os.path.join(os.path.dirname(__file__), '../ThirdParty/ROUGE/ROUGE-1.5.5/')
        self.pyrouge_dir = '/cluster/home/qimengnan/HMNet/ThirdParty/ROUGE/ROUGE-1.5.5/'

        self.eval_batches_num = self.opt.get('EVAL_BATCHES_NUM', float('Inf'))
        self.best_score = float("Inf")
        self.best_res = {}

    def reset_best_score(self, set_high=True):
        if set_high:
            self.best_score = float("Inf")
        else:
            self.best_score = -float("Inf")

    def make_html_safe(self, s):
        s = s.replace("<", "&lt;")
        s = s.replace(">", "&gt;")
        return s

    def print_to_rouge_dir(self, summaries, dir, suffix, split_chars, special_char_dict=None):
        for idx, summary in enumerate(summaries):
            fname = os.path.join(dir, "%06d_%s.txt" % (idx, suffix))
            with open(fname, "wb") as f:
                sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summary)
                for i, sent in enumerate(sents):
                    if split_chars:
                        # sent = re.sub(r'([\u4e00-\u9fff])', r' \1 ', sent)
                        for x in re.finditer( r'([\u4e00-\u9fff])', sent):
                            if not x.group(1) in special_char_dict:
                                special_char_dict[x.group(1)] = len(special_char_dict)
                            sent = sent.replace(x.group(1), ' {} '.format(special_char_dict[x.group(1)]))
                    if i == len(sents) - 1:
                        to_print = sent.encode('utf-8')
                    else:
                        to_print = sent.encode('utf-8') + '\n'.encode('utf-8')
                    f.write(to_print)

    def print_to_rouge_dir_gt(self, summaries, dir, suffix, split_chars):
        if split_chars:
            char_dict = {}

        for idx, summary in enumerate(summaries):
            for ref_idx, sub_summary in enumerate(summary.split(' ||| ')):
                fname = os.path.join(dir, "%s.%06d_%s.txt" % (ascii_uppercase[ref_idx], idx, suffix))
                with open(fname, "wb") as f:
                    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', sub_summary)
                    for i, sent in enumerate(sents):
                        if split_chars:
                            for x in re.finditer( r'([\u4e00-\u9fff])', sent):
                                if not x.group(1) in char_dict:
                                    char_dict[x.group(1)] = len(char_dict)
                                sent = sent.replace(x.group(1), ' {} '.format(char_dict[x.group(1)]))

                        if i == len(sents) - 1:
                            to_print = sent.encode('utf-8')
                        else:
                            to_print = sent.encode('utf-8') + '\n'.encode('utf-8')
                        f.write(to_print)

        if split_chars:
            return char_dict

    # def filter_empty(self, predictions, groundtruths):
    #     new_predicitons = []
    #     new_groundtruths = []
    #
    #     for pred, gt in zip(predictions, groundtruths):
    #         if len(gt) == 0:
    #             continue
    #         new_groundtruths.append(gt)
    #         if len(pred) == 0:
    #             new_predicitons.append('<ept>')
    #         else:
    #             new_predicitons.append(pred)
    #     return new_predicitons, new_groundtruths

    def _convert_tokens_to_string(self, tokenizer, tokens):
        if 'EVAL_TOKENIZED' in self.opt:
            tokens = [t for t in tokens if t not in tokenizer.all_special_tokens]
        if 'EVAL_LOWERCASE' in self.opt:
            tokens = [t.lower() for t in tokens]
        if 'EVAL_TOKENIZED' in self.opt:
            return ' '.join(tokens)
        else:
            return tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)

    def eval_batches(self, module, dev_batches, save_folder, label=''):
        #max_sent_len = int(self.opt['MAX_GEN_LENGTH'])

        logger.info('Saving folder is {}'.format(save_folder))

        predictions = [] # prediction of tokens from model
        x_ids = [] # input token ids
        y_ids = [] # groundtruths token ids
        gts = [] # groundtruths
        got_better_score = False
        eval_loss = 0.0
        nb_eval_steps = 0

        with torch.no_grad():
            for j, dev_batch in enumerate(dev_batches):
                for b in dev_batch:
                    if torch.is_tensor(dev_batch[b]):
                        dev_batch[b] = dev_batch[b].to(self.opt['device'])

                topic_segment_output = module(dev_batch, crf_decode=True)
                pred = topic_segment_output[1].squeeze(0)
                gt = dev_batch['encoder_input_topics']
                predictions.extend([pred])
                gts.extend([gt])
        # use MPI to gather results from all processes / GPUs
        # the result of the gather operation is a list of sublists
        # each sublist corresponds to the list created on one of the MPI processes (or GPUs, respectively)
        # we flatten this list into a "simple" list
        assert len(predictions) == len(gts), "len(predictions): {0}, len(gts): {1}".format(len(predictions), len(gts))
        comm = MPI.COMM_WORLD
        predictions = comm.gather(predictions, root=0)
        # if GPU numbers are high (>=8), passing x_ids, y_ids to a rank 0 will cause out of memory
        # x_ids = comm.gather(x_ids, root=0)
        # y_ids = comm.gather(y_ids, root=0)
        gts = comm.gather(gts, root=0)
        if self.opt['rank'] == 0:
            # flatten lists
            predictions = [item for sublist in predictions for item in sublist]

            gts = [item for sublist in gts for item in sublist]
            # import pdb; pdb.set_trace()
            assert len(predictions) == len(gts), \
                "len(predictions): {0}, len(gts): {1}".format(len(predictions), len(gts))

            # write intermediate results only on rank 0
            #top_1_predictions = [pred[0] for pred in predictions]


            try:
                result = self.eval(predictions, gts)
            except Exception as e:
                logger.exception("Eval ERROR")
                result = {}
                score = -float("Inf")
                pass # this happens when no overlapping between pred and gts
            else:
                score = (result['pk'] + result['windiff'])/2
                if score < self.best_score:

                    self.best_score = score
                    self.best_res = result
                    got_better_score = True

        else:
            result = {}
            score = -float("Inf")
            got_better_score = False

        return result, score, got_better_score

    def eval(self, predictions, groundtruths):
        # predictions, groundtruths = self.filter_empty(predictions, groundtruths)
        #predictions = [self.make_html_safe(w) for w in predictions]
        #groundtruths = [self.make_html_safe(w) for w in groundtruths]
        pred_dir = os.path.join(self.save_dir, "predictions")
        if os.path.exists(pred_dir):
            shutil.rmtree(pred_dir)
        os.makedirs(pred_dir)

        gt_dir = os.path.join(self.save_dir, "groundtruths")
        if os.path.exists(gt_dir):
            shutil.rmtree(gt_dir)
        os.makedirs(gt_dir)

        acc = accuracy.Accuracy()
        for i in range(len(predictions)):
            output_seg = predictions[i].squeeze(0).cpu().numpy()[1:]
            target_seg = groundtruths[i].squeeze(0).cpu().numpy()[1:]
            prd = np.append(output_seg, [1])
            gt = np.append(target_seg, [1])
            #print('prd:',prd)
            #print('gt:',gt)
            acc.update(prd, gt)
        pk, windiff = acc.calc_accuracy()

        results = {
            'pk': pk,
            'windiff': windiff
        }
        return results
