import os
import jsonlines
import spacy
import json
from tqdm import tqdm
from nltk.corpus import stopwords
from spacy.tokens.span import Span
import numpy as np
from random import Random, shuffle, random
import torch
import math
from DataLoader import iterators
import json
import struct
from timeit import default_timer as timer
from ThirdParty.Huggingface.Transformers.src.transformers import tokenization_transfo_xl
from Models.Networks.MeetingNet_Transformer_ernie_cpu import MeetingNet_Transformer_ernie_cpu
from Utils.Arguments import Arguments
import gradio

nlp = spacy.load('en_core_web_lg', parser = False)
POS = {w: i for i, w in enumerate([''] + list(nlp.tagger.labels))}
ENT = {w: i for i, w in enumerate([''] + nlp.entity.move_names)}


max_transcript_len = 8300
max_sentence_len = 30
max_sentence_num = 400

role_dict_file = '/cluster/home/qimengnan/HMNet/ExampleRawData/meeting_summarization/role_dict_ms.json'
role_dict = json.load(open(role_dict_file))
print("role_dict:", role_dict)
anon_roles = False # whether to convert all speakers to speaker-0, speaker-1, ...

# load tokenizer
tokenizer_class = getattr(tokenization_transfo_xl, 'TransfoXLTokenizer')
pretrained_tokenizer_path = 'ExampleInitModel/transfo-xl-wt103'

# here is a simple workaround to make sure all special tokens are not None
config = {}
tokenizer = tokenizer_class.from_pretrained(pretrained_tokenizer_path)
special_tokens_tuple_list = [("eos_token", 128), ("unk_token", 129), ("pad_token", 130), ("bos_token", 131)]

for special_token_name, special_token_id_offset in special_tokens_tuple_list:
    if getattr(tokenizer, special_token_name) == None:
        setattr(tokenizer, special_token_name,
                tokenizer.convert_ids_to_tokens(len(tokenizer) - special_token_id_offset))
        config[special_token_name] = tokenizer.convert_ids_to_tokens(
            len(tokenizer) - special_token_id_offset)
        config[special_token_name + '_id'] = len(tokenizer) - special_token_id_offset

# stopword = stopwords.words('english')

def preprocess(dialog_str):
    utt = []
    dialog_list = dialog_str.split('\n')
    print('dialog_list:', dialog_list)
    for dialog in dialog_list:
        utt.append({"speaker": dialog.split(':', 1)[0].strip(), "utt": dialog.split(':', 1)[1].strip()})
    return utt

def tokenization(utt):

    meeting_list = []
    utterance_list = []
    sample_role_dict = {}
    #global vocb
    for i in range(len(utt)):
        text = utt[i]['utt']
        doc = nlp(text)
        tokens = []
        pos_ids = []
        ent_ids = []
        #utt_list = []
        banned_word = [' ', 'um', 'Um', 'hm', 'Hmm', 'hmm', '-', "'",'mm', 'Mm', 'MM', 'Kay', 'kay']
        puncuation = ['.', '?', ',', '!', ':', ';', '...', '|', "'"]
    # -----------utt--------------------------
        for token in doc:
            if POS[token.tag_] != 37 and (len(str(token)) !=1 or str(token) in puncuation or str(token) == 'a' or str(token) == 'A' or str(token) == 'i' or str(token) == 'I') and str(token) not in banned_word:
                tokens.append(str(token))
                pos_ids.append(POS[token.tag_])
                #vocb.add(str(token))

                if token.ent_type_ != '':
                    move_name = str(token.ent_iob_) + '-' + str(token.ent_type_)
                    ent_id = ENT[move_name]
                elif token.ent_iob_ == 'O':
                    ent_id = ENT[token.ent_iob_]
                else:
                    raise ValueError('Invalid token')
                ent_ids.append(ent_id)

        if tokens:
            if utt[i]['speaker'] not in role_dict:
                if utt[i]['speaker'] not in sample_role_dict:
                    sample_role_dict[utt[i]['speaker']] = len(sample_role_dict)
                #"cnn-{}".format(sample_role_dict[utt[i]['speaker']])
                utt_role = role_dict.get('cnn-{}'.format(sample_role_dict[utt[i]['speaker']]), 0)
            else:
                utt_role = role_dict.get(utt[i]['speaker'], 0)
            """
            turn = ' '.join(tokens)
            utt_list.append({
                "word": turn,
                "pos_id": pos_ids,
                "ent_id": ent_ids
                })
            """
            # -----------------meeting------------------------------
            utterance_list.append({"speaker": utt[i]['speaker'], "role": utt_role, "utt": {"word": tokens, "pos_id": pos_ids, "ent_id": ent_ids}})
    meeting_list.append(utterance_list)

    return meeting_list

    # use SamplingRandomMapIterator because it applies one-to-one mapping (new iterator take one document from source iterator, apply transform, and output it) with checkpointed random state

def pad_batch(batch):
    # padding and generate final batch
    x_sent_batch = []
    x_role_batch = []
    x_pos_batch = []
    x_ent_batch = []
    y_sent_batch = []

    encoder_tokens, decoder_tokens = [], []

    for datum in batch:
        x_sent = []
        x_role = []
        x_pos = []
        x_ent = []

        sample_input_tokens = []

        total_word_len = 0
        total_sent_len = 0

        #assert len(datum['meeting']) > 0
        for m in datum:  # each m is actually a turn
            words = m['utt']['word']
            pos = m['utt']['pos_id']
            ent = m['utt']['ent_id']
            L = len(words)
            # assert L < max_transcript_len, "a turn {} is longer than max_transcript_len".format(' '.join(words))
            if L > max_transcript_len:
                # this is rarely happpened when a turn is super long
                # in this case we just skip it to save memory
                continue
            if total_word_len + L > max_transcript_len or total_sent_len + 1 > max_sentence_num:
                break

            sample_input_tokens.extend(words)

            for i in range(math.ceil(L / max_sentence_len)):
                x_role.append(m['role'])
                sub_words = words[i * max_sentence_len:min((i + 1) * max_sentence_len, L)]
                x_sent.append([tokenizer.bos_token] + sub_words + [tokenizer.eos_token])
                x_pos.append([0] + pos[i * max_sentence_len:min((i + 1) * max_sentence_len, L)] + [0])
                x_ent.append([0] + ent[i * max_sentence_len:min((i + 1) * max_sentence_len, L)] + [0])

                total_sent_len += 1

            total_word_len += L

        '''
        if is_train:  # training
            y_sent = [tokenizer.bos_token] + datum['target']['sequence'][:max_gen_length] + [tokenizer.eos_token]
        else:
            y_sent = [tokenizer.bos_token] + datum['target']['sequence'] + [tokenizer.eos_token]
        '''

        if len(x_sent) > 0:
            # this could be false when there is a single but very long turn
            x_sent_batch.append(x_sent)
            x_role_batch.append(x_role)
            x_pos_batch.append(x_pos)
            x_ent_batch.append(x_ent)
            #y_sent_batch.append(y_sent)

            encoder_tokens.append(sample_input_tokens)
            #decoder_tokens.append(y_sent)

    if len(x_sent_batch) == 0:
        # this could happen when there is a single but very long turn
        # leading the whole batch with all instances filtered
        return None

    # count max length
    x_max_doc_len = max([len(s) for s in x_sent_batch])
    x_max_sent_len = max([max([len(t) for t in s]) for s in x_sent_batch])
    #y_max_len = max([len(s) for s in y_sent_batch])
    x_role_max_len = max([len(s) for s in x_role_batch])
    actual_size = len(x_sent_batch)

    actual_tokens_per_batch = actual_size * (x_max_doc_len * x_max_sent_len)


    # create tensors
    x_tensor = torch.LongTensor(actual_size, x_max_doc_len, x_max_sent_len).fill_(tokenizer.pad_token_id)
    x_pos_tensor = torch.LongTensor(actual_size, x_max_doc_len, x_max_sent_len).fill_(0)
    x_ent_tensor = torch.LongTensor(actual_size, x_max_doc_len, x_max_sent_len).fill_(0)
    x_role_tensor = torch.LongTensor(actual_size, x_role_max_len).fill_(0)
    #y_tensor = torch.LongTensor(actual_size, y_max_len).fill_(tokenizer.pad_token_id)

    for i in range(len(x_sent_batch)):
        for j in range(len(x_sent_batch[i])):
            x_tensor[i, j, :len(x_sent_batch[i][j])] = torch.LongTensor(
                tokenizer.convert_tokens_to_ids(x_sent_batch[i][j]))
            #y_tensor[i, :len(y_sent_batch[i])] = torch.LongTensor(tokenizer.convert_tokens_to_ids(y_sent_batch[i]))

        for j in range(len(x_pos_batch[i])):
            x_pos_tensor[i, j, :len(x_pos_batch[i][j])] = torch.LongTensor(x_pos_batch[i][j])
        for j in range(len(x_ent_batch[i])):
            x_ent_tensor[i, j, :len(x_ent_batch[i][j])] = torch.LongTensor(x_ent_batch[i][j])

        x_role_tensor[i, :len(x_role_batch[i])] = torch.LongTensor(x_role_batch[i])

    return {
        'encoder_input_ids': x_tensor,
        'encoder_input_roles': x_role_tensor,
        'encoder_input_pos': x_pos_tensor,
        'encoder_input_ent': x_ent_tensor,
        'decoder_input_ids': x_role_tensor,
    }

def convert_tokens_to_string(tokenizer, tokens):
    tokens = [t for t in tokens if t not in tokenizer.all_special_tokens]
    tokens = [t.lower() for t in tokens]
    return ' '.join(tokens)

def eval_batches(opt, module, dev_batches):

    max_sent_len = int(opt['MAX_GEN_LENGTH'])
    predictions = [] # prediction of tokens from model

    with torch.no_grad():
        for j, dev_batch in enumerate(dev_batches):
            beam_search_res = module(dev_batch, beam_search=True, max_sent_len=max_sent_len)
            pred = [[t[0] for t in x] if len(x) > 0 else [[]] for x in beam_search_res]
            predictions.extend([[convert_tokens_to_string(tokenizer, tt) for tt in t] for t in pred])
    return predictions

# ------------main()-----------------
#if __name__ == '__main__':
def meeting_sum(input_dialog, model):
    #data = input("input:")
    #data = open(input_file,'r').read()
    data = input_dialog
    print('data:', data)
    conf_file = 'ExampleConf/conf_eval_ernie_AMI'
    conf_args = Arguments(conf_file)
    opt = conf_args.readArguments()
    opt['cuda'] = False
    opt['confFile'] = conf_file
    if 'datadir' not in opt:
        opt['datadir'] = os.path.dirname(conf_file)  # conf_file specifies where the data folder is
    opt['basename'] = os.path.basename(conf_file)  # conf_file specifies where the name of save folder is
    opt['command'] = 'evaluate'

    if model == "AMI-fintuned":
        opt['PYLEARN_MODEL'] = '../ExampleInitModel/AMI_Finetuned'
    elif model == "ICSI-fintuned":
        opt['PYLEARN_MODEL'] = '../ExampleInitModel/ICSI_Finetuned'
    elif model == "pretrained":
        opt['PYLEARN_MODEL'] = '../ExampleInitModel/Pretrained'
    else:
        opt.pop('PYLEARN_MODEL')
    print('opt:', opt)

    utterance = preprocess(data)
    #print("utterance:", utterance)
    meeting_list = tokenization(utterance)
    #print('meeting_list:', meeting_list)
    input_batch = pad_batch(meeting_list)
    #print("input_batch:", input_batch)
    eval_batch = [input_batch]
    module = MeetingNet_Transformer_ernie_cpu(opt)
    predictions = eval_batches(opt, module, eval_batch)
    return predictions[0][0]

#input_file = gradio.inputs.File(file_count="single", type="file", label=None, optional=False)
input_dialog = gradio.inputs.Textbox(lines=400, type="str", label="input")
output_text = gradio.outputs.Textbox(type="str", label="output")
title = "QMN的英文会议摘要生成模型"
description = "左边输入符合对话标准的格式（speaker:utterance）作为输入，最后右边会输出对应的会议对话摘要。"
gradio.Interface(fn=meeting_sum, inputs=[input_dialog, gradio.inputs.Dropdown(choices=["AMI-fintuned", "ICSI-fintuned", "pretrained"], type="value", label="model")],outputs=output_text, examples=None, title=title, description=description, capture_session=True).launch(share=True)