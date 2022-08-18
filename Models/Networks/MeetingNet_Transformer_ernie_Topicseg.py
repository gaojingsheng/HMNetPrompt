# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import math
import numpy as np
import random
import json
import time
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from Models.Networks.Layers import dropout, set_seq_dropout
#from transformers import GPT2ForSequenceClassification,GPT2Config,GPT2Model
from Models.Networks.Transformer import EncoderBlock, LayerNorm, Embedder, Splitter, Attention, MLP, TurnLeverEmbedder
#from Models.Networks.Transformer_cpu import EncoderBlock, LayerNorm, Embedder, Splitter, Attention, MLP
from Models.knowledge_bert.modeling_rolemask import BertForPreTraining, BertConfig
from ThirdParty.Huggingface.Transformers.src.transformers import tokenization_transfo_xl 
from ThirdParty.Huggingface.Transformers.src.transformers.modeling_encoder_decoder import calc_banned_ngram_tokens, calc_banned_bad_words_ids, top_k_top_p_filtering, BeamHypotheses

from Models.Networks.crf import CRF
import sys
import os

import spacy
nlp = spacy.load('en_core_web_lg', parser = False)
POS = {w: i for i, w in enumerate([''] + list(nlp.tagger.labels))}
ENT = {w: i for i, w in enumerate([''] + nlp.entity.move_names)}

class MeetingNet_Transformer_ernie_Topicseg(nn.Module):
    def __init__(self, opt):
        super(MeetingNet_Transformer_ernie_Topicseg, self).__init__()

        self.opt = opt
        self.use_cuda = (self.opt['cuda'] == True)
        self.config = {}

        # load tokenizer
        self.tokenizer_class = getattr(tokenization_transfo_xl, opt['PRE_TOKENIZER'])
        self.pretrained_tokenizer_path = os.path.join(opt['datadir'], opt['PRE_TOKENIZER_PATH'])
        if not os.path.isdir(self.pretrained_tokenizer_path):
            '''
            This if-else statement makes sure the pre-trained tokenizer exists
            If it does not exist, it assumes the input string is the HuggingFace tokenizer name,
            and downloads it from their website.
            '''
            self.pretrained_tokenizer_path = opt['PRE_TOKENIZER_PATH']
        else:
            print('Loading Tokenizer from {}...'.format(self.pretrained_tokenizer_path))

        # here is a simple workaround to make sure all special tokens are not None
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_tokenizer_path)
        special_tokens_tuple_list = [("eos_token", 128), ("unk_token", 129), ("pad_token", 130), ("bos_token", 131)]

        for special_token_name, special_token_id_offset in special_tokens_tuple_list:
            if getattr(self.tokenizer, special_token_name) == None:
                setattr(self.tokenizer, special_token_name, self.tokenizer.convert_ids_to_tokens(len(self.tokenizer)-special_token_id_offset))
                self.config[special_token_name] = self.tokenizer.convert_ids_to_tokens(len(self.tokenizer)-special_token_id_offset)
                self.config[special_token_name+'_id'] =  len(self.tokenizer)-special_token_id_offset

        self.vocab_size = self.tokenizer.vocab_size
        opt['vocab_size'] = self.vocab_size
        self.role_size = int(opt['ROLE_SIZE'])
        vocab_dim = int(opt['VOCAB_DIM'])
        role_dim = int(opt['ROLE_DIM'])
        opt['transformer_embed_dim'] = vocab_dim
        embed = nn.Embedding(self.vocab_size, vocab_dim, padding_idx=self.tokenizer.pad_token_id)
        nn.init.normal_(embed.weight, std=0.02)
        embedder = Embedder(opt, embed)
        #turnleverembedder = TurnLeverEmbedder(opt)
        role_embed = nn.Embedding(self.role_size, role_dim, padding_idx=0)

        self.encoder = Encoder(opt, self.vocab_size, vocab_dim, role_dim, embedder, role_embed)

        if 'PYLEARN_MODEL' in self.opt:
            self.from_pretrained(os.path.join(opt['datadir'], opt['PYLEARN_MODEL']))

    def save_pretrained(self, save_dir):
        network_state = dict([(k, v) for k, v in self.state_dict().items()])
        params = {
            'state_dict': {'network': network_state},
            'config': self.opt,
        }
        torch.save(params, os.path.join(save_dir, 'model.pt'))

    def from_pretrained(self, load_dir):
        checkpoint = torch.load(os.path.join(load_dir, 'model.pt'), map_location=torch.device('cuda', self.opt['local_rank']))
        #checkpoint = torch.load(os.path.join(load_dir, 'model.pt'), map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
        
        self.load_state_dict(state_dict['network'])

        return self

    def get_training_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, batch, crf_decode=None):
        if crf_decode is not None:
            # return self.beam_search(batch, max_sent_len)
            return self.generate(batch, crf_decode)

        topic_segment_output, sent_encoder_outputs = self._forward(**batch)


        # assume all encoder-decoder model input has BOS and EOS
        # otherwise the loss will be ill-defined
        #return vocab_logprob, topic_segment_output, role_ae_loss
        return topic_segment_output, sent_encoder_outputs

    '''
        Input:
         encoders_input_ids = 1 * num_turns * x_len (word_ids)
         encoders_input_roles = 1 * num_turns (role_ids)
         encoders_input_pos = 1 * num_turns * x_len (pos_ids)
         encoders_input_ent = 1 * num_turns * x_len (ent_ids)
         decoder_input_ids = 1 * y_len (word_ids) 
        Output:
          vocab_logprob  = 1 x y_len x vocab_size
    '''
    def _forward(self, **kwargs):
        
        encoder_input_ids = kwargs.pop('encoder_input_ids')
        encoder_input_roles = kwargs.pop('encoder_input_roles')
        encoder_input_pos = kwargs.pop('encoder_input_pos')
        encoder_input_ent = kwargs.pop('encoder_input_ent')
        encoder_input_topics = kwargs.pop('encoder_input_topics')
        encoder_input_topics_mask = kwargs.pop('encoder_input_topics_mask')

        token_encoder_outputs, sent_encoder_outputs, topic_segment_output = self.encoder(encoder_input_ids, encoder_input_roles, encoder_input_pos, encoder_input_ent, encoder_input_topics, encoder_input_topics_mask)
        return topic_segment_output, sent_encoder_outputs

    def generate(self, batch, crf_decode):
        self.eval()

        input_ids = batch["encoder_input_ids"]
        input_roles = batch["encoder_input_roles"]
        input_pos = batch["encoder_input_pos"]
        input_ent = batch["encoder_input_ent"]
        input_topics = batch['encoder_input_topics']
        input_topics_mask = batch['encoder_input_topics_mask']

        token_encoder_outputs, sent_encoder_outputs, topic_segment_output = self.encoder(input_ids,
                                                                                         input_roles,
                                                                                         input_pos,
                                                                                         input_ent,
                                                                                         input_topics, input_topics_mask, crf_decode)
        return topic_segment_output

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if type(past) is tuple:
            encoder_outputs = past
        else:
            encoder_outputs = (past,)

        return {
            "decoder_input_ids": input_ids,
            "token_encoder_outputs": encoder_outputs[0],
            "sent_encoder_outputs": encoder_outputs[1],
        }

    def prepare_scores_for_generation(self, scores, **kwargs):
        return scores

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

    @torch.no_grad()
    def _generate(
        self,
        input_ids=None,
        input_roles=None,
        input_pos=None,
        input_ent=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=False,
        num_beams=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
    ):

        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 3, "Input prompt should be of shape (batch_size, sequence length)."

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        vocab_size = self.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample

        effective_batch_size = batch_size
        effective_batch_mult = 1

        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        assert (
            decoder_start_token_id is not None
        ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"

        encoder_outputs = self.encoder(input_ids, input_roles, input_pos, input_ent)

        # # Expand input ids if num_beams > 1 or num_return_sequences > 1
        # if num_return_sequences > 1 or num_beams > 1:
        #     input_sent_len = input_ids.shape[2]
        #     input_word_len = input_ids.shape[3]
        #     input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_sent_len, input_word_len)
        #     attention_mask = attention_mask.unsqueeze(1).expand(
        #         batch_size, effective_batch_mult * num_beams, input_sent_len, input_word_len
        #     )

        #     input_ids = input_ids.contiguous().view(
        #         effective_batch_size * num_beams, input_sent_len, input_word_len
        #     )  # shape: (batch_size * num_return_sequences * num_beams, input_sent_len, input_word_len)
        #     attention_mask = attention_mask.contiguous().view(
        #         effective_batch_size * num_beams, input_sent_len, input_word_len
        #     )  # shape: (batch_size * num_return_sequences * num_beams, input_sent_len, input_word_len)

        # create empty decoder_input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        cur_len = 1

        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), encoder_outputs[1].index_select(0, expanded_batch_idxs))


        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        batch_size,
        encoder_outputs,
        attention_mask,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask)
            
            outputs = self.decoder(**model_inputs)
            next_token_logits = outputs[:, -1, :]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                self.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                next_token_logits[:, eos_token_id] = -float("inf")

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            cur_len = cur_len + 1

        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids

        for hypo_idx, hypo in enumerate(input_ids):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        return decoded


    # force one of token_ids to be generated by setting prob of all other tokens to 0.
    def _force_token_ids_generation(self, scores, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` and `mems` is at 2nd position
            reordered_layer_past = [layer_past[i, :].unsqueeze(0).clone().detach() for i in beam_idx]
            reordered_layer_past = torch.cat(reordered_layer_past, dim=0)
            # check that shape matches
            assert reordered_layer_past.shape == layer_past.shape
            reordered_past.append(reordered_layer_past)
        past = tuple(reordered_past)
        return past


'''
  Transformer encoder
'''
class MeetingTransformerEncoder(nn.Module):
    '''
      Input:
        transformer_embed_dim: transformer dimension
    '''
    def __init__(self, opt, transformer_embed_dim):
        super(MeetingTransformerEncoder, self).__init__()
        vocab = int(opt['vocab_size'])
        n_layer = int(opt['WORLDLEVEL_LAYER'])
        opt['transformer_embed_dim'] = transformer_embed_dim
        block = EncoderBlock(opt)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])

    '''
      Input:
        x: batch x len x n_state
      Output:
        h: batch x len x n_state
    '''
    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h, None)
        return h
'''
Turn-Level Encoder
'''
class TurnLevelEncoder(nn.Module):
    '''
      Input:
        transformer_embed_dim: transformer dimension
    '''
    def __init__(self, opt, transformer_embed_dim):
        super(TurnLevelEncoder, self).__init__()
        #vocab = int(opt['vocab_size'])
        #n_layer = int(opt['TURNLEVEL_LAYER'])
        config=BertConfig(opt['ERNIECONFIG_PATH'])
        config.hidden_size = transformer_embed_dim
        config.vocab_size = int(opt['vocab_size'])
        config.intermediate_size = 4*transformer_embed_dim
        #print('ERNIECONFIG:',config)
        self.opt = opt
        self.turn_level_encoder= BertForPreTraining(config)
    '''
      Input:
        x: batch x len x n_state
      Output:
        h: batch x len x n_state
    '''
    def forward(self, x, x_role, role_embed):
        uniq_idx = np.unique(x_role.cpu().numpy())
       # print('x_role:',x_role)
        #print('uniq_idx:',uniq_idx)
        ent_candidate = role_embed(torch.LongTensor(uniq_idx + 1).to(x.device))
        # build entity labels
        d = {}
        dd = []
        for i, idx in enumerate(uniq_idx):
            d[idx] = i
            dd.append(idx)
        ent_size = len(uniq_idx) - 1

        '''def map(x):
            if x == -1:
                return -1
            else:
                rnd = random.uniform(0, 1)
                if rnd < 0.05:
                    return dd[random.randint(1, ent_size)]
                elif rnd < 0.2:
                    return -1
                else:
                    return x'''

        ent_labels = x_role.cpu()
        d[-1] = -1
        ent_labels = ent_labels.apply_(lambda x: d[x])
        ent_labels = ent_labels.to(x.device)
        
        if 'USE_ENTMASK' in self.opt:
            mask = x_role.cpu()
            masked_indices = torch.bernoulli(torch.full(mask.shape, 0.15)).bool()
            #x_role.apply_(map)
            #ent_emb = embed(entity_idx + 1)
            mask[~masked_indices] = -1  # We only compute loss on masked tokens
            indices_replaced = torch.bernoulli(torch.full(mask.shape, 0.8)).bool() & masked_indices
            x_role[indices_replaced] = 20  #tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            # 10% of the time, we replace masked input tokens with random word
            """
            对于mask_indices剩下的20% 在进行提取,取其中一半进行random 赋值,剩下一半保留原来值. 
            """
            indices_random = torch.bernoulli(torch.full(mask.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(21, 46, mask.shape, dtype=torch.long).to(x.device)
            x_role[indices_random] = random_words[indices_random]
            ent_emb = role_embed(x_role)
    
            #mask.apply_(map)
            mask.apply_(lambda y: 0 if y == -1 else 1)
            mask[:, 0] = 1
            mask = mask.to(x.device)
        else:
            ent_emb = role_embed(x_role)
            mask = torch.ones_like(x_role)
            
        sequence_output, role_ae_loss = self.turn_level_encoder(input_embedding=x, token_type_ids=None, attention_mask=None, input_ent=ent_emb, ent_mask=mask, candidate=ent_candidate, ent_labels=ent_labels)
        if 'USE_ENTMASK' not in self.opt:
            role_ae_loss = 0*role_ae_loss
        #print('sequence_output:',sequence_output.device)
       # print('role_ae_loss:',role_ae_loss.device)
        return sequence_output, role_ae_loss

'''
    TopicSegment module use Linear and CRF
'''
class TopicSegment(nn.Module):
    def __init__(self, opt, transformer_embed_dim, vocab_size):
        super(TopicSegment, self).__init__()
        self.opt = opt
        self.dropout = nn.Dropout(self.opt['DROPOUT'])
        self.classifier = nn.Linear(transformer_embed_dim, 2)
        self.crf = CRF(num_tags=2, batch_first=True)
        #self.init_weights()

    def forward(self, x, labels, labels_mask, crf_decode=None):
        #hidden_states = self.gpt(inputs_embeds=x)[0]
        x = self.dropout(x)
        logits = self.classifier(x)
        if crf_decode is not None:
            eval_loss = self.crf(emissions=logits, tags=labels, mask=labels_mask)
            tags = self.crf.decode(logits, mask=labels_mask, nbest=1)
            outputs = (tags,)
            outputs = (-1 * eval_loss,) + outputs
        else:
            loss = self.crf(emissions=logits, tags=labels, mask=labels_mask)
            outputs = (logits,)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores

class Encoder(nn.Module):
    '''
     vocab_size: size of input vocabulary
     embed_size: word embedding dimension of dictionary
     role_dim: role embedding dimension
     embed: the nn.Embedding for vocab
     role_embed: the nn.Embedding for role
    '''
    def __init__(self, opt, vocab_size, embed_size, role_dim, embedder, role_embed):
        super(Encoder, self).__init__()
        self.opt = opt
        self.vocab_size = vocab_size

        set_seq_dropout('VARIATIONAL_DROPOUT' in self.opt)

        self.embed_size = embed_size
        self.embedder = embedder
        #self.turnleverembedder = TurnLeverEmbedder(self.opt)
        self.role_embed = role_embed

        self.token_transformer_dim = embed_size
        if 'USE_POSENT' in opt:
            print('Use POS and ENT')
            pos_dim = opt['POS_DIM']
            ent_dim = opt['ENT_DIM']
            self.pos_embed = nn.Embedding(len(POS), pos_dim)
            self.ent_embed = nn.Embedding(len(ENT), ent_dim)
            self.token_transformer_dim += pos_dim + ent_dim

        self.sent_transformer_dim = self.token_transformer_dim
        if 'USE_ROLE' in opt:
            print("USE_ROLE")
            role_dim = opt['ROLE_DIM']
            self.sent_transformer_dim += role_dim

        self.token_encoder = MeetingTransformerEncoder(opt, self.token_transformer_dim)
        #self.sent_encoder = MeetingTransformerEncoder(opt, self.sent_transformer_dim)
        self.sent_encoder = TurnLevelEncoder(opt, self.sent_transformer_dim)
        self.topic_segment = TopicSegment(opt, self.token_transformer_dim, self.vocab_size)

    '''
     x = bz * sent_num * x_len (word_ids)
     x_role = bz * sent_num (role_ids)
     x_pos = bz * sent_num * x_len (pos_ids)
     x_ent = bz * sent_num * x_len (ent_ids)
     outputs:
       token_encoder_outputs: bz x x_len_total x token_transformer_dim
       sent_encoder_outputs:  bz x sent_num x sent_transformer_dim
    '''
    def forward(self, x, x_role, x_pos, x_ent, topic_labels, topics_mask, crf_decode=None):
        batch_size = x.size(0)
        sent_num = x.size(1)
        x_len = x.size(2)

        # x contains word id >= vocab_size
        vocab_x = x.clone()
        vocab_x[vocab_x >= self.vocab_size] = 1 # UNK
        #embedded = self.embedder(vocab_x.view(batch_size, -1))
        embedded = self.embedder(vocab_x.view(-1, x_len))
        # embedded = 1 x sent_num * x_len x embed_size
        embedded = embedded.view(batch_size, sent_num, x_len, -1)
        # embedded = 1 x sent_num x x_len x embed_size

        if 'USE_ROLE' in self.opt:
            role_embed = self.role_embed(x_role) # 1 x sent_num x role_dim

        if 'USE_POSENT' in self.opt:
            embedded = torch.cat([embedded, self.pos_embed(x_pos), self.ent_embed(x_ent)], dim=3)
            # 1 x sent_num x x_len x (embed_size + pos_dim + ent_dim )

        feat_dim = embedded.size(3)

        token_transformer_output = self.token_encoder(embedded.view(-1, x_len, feat_dim))
        token_transformer_dim = token_transformer_output.size(2)
        token_transformer_output = token_transformer_output.view(batch_size, sent_num, x_len, token_transformer_dim)
        # 1 x sent_num x x_len x token_transformer_dim

        sent_encoder_inputs = token_transformer_output[:, :, 0, :] # 1 x sent_num x token_transformer_dim


        #topic_segment_output = None
        #sent_encoder_inputs = self.turnleverembedder(sent_encoder_inputs)  # 在这里重新添加位置向量（体现不同说话人的位置信息）

        if 'USE_ROLE' in self.opt:
            sent_encoder_inputs = torch.cat([sent_encoder_inputs, role_embed], dim=2)
        #sent_encoder_outputs = self.sent_encoder(sent_encoder_inputs) # 1 x sent_num x sent_transformer_dim
        sent_encoder_outputs, role_ae_loss = self.sent_encoder(sent_encoder_inputs, x_role, self.role_embed)

        # 在这里插入主题分割的模块
        topic_segment_output = self.topic_segment(sent_encoder_outputs, topic_labels, topics_mask, crf_decode)

        return token_transformer_output, sent_encoder_outputs, topic_segment_output




    