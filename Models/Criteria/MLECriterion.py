# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLECriterion(nn.Module):
    '''
        Class to define loss give input, model output and groundtruth
    '''

    def __init__(self, opt, module):
        super().__init__()
        self.opt = opt
        self.ignore_index = self.opt['IGNORE_INDEX'] if 'IGNORE_INDEX' in self.opt \
                            else module.tokenizer.pad_token_id

    def forward(self, vocab_logprob, batch):
        extended_vocab_size = vocab_logprob.shape[2]
        y = batch["decoder_input_ids"]

        if 'USE_BOS_TOKEN' in self.opt:
            y = y[:, 1:]

        if 'USE_EOS_TOKEN' in self.opt:
            vocab_logprob = vocab_logprob[:, :-1, :]

        loss = F.nll_loss(vocab_logprob.contiguous().view(-1, extended_vocab_size), y.contiguous().view(-1), ignore_index=self.ignore_index)

        return loss

class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, batch):
        targets = batch['encoder_input_topics']
        ts = torch.zeros(targets.size(0), targets.size(1), 2).to(inputs.device)
        for i in range(targets.size(0)):
            for j in range(targets.size(1)):
                ts[i,j,targets[i,j]] = 1
        #print('targets:',ts)
        #print('inputs:',inputs)
        #inputs.retain_grad()
        #print('topic_seg.grad:',inputs.grad)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs.float().requires_grad_(True), ts.float(), reduction='none')
        #peint('BCE_loss:',BCE_loss)
        ts = ts.type(torch.long)
        bs = ts.size(0)
#         at = self.alpha.gather(0, targets.data)
        a = 2*self.alpha-1
        b = 1- self.alpha
        at = ts*a + b
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
