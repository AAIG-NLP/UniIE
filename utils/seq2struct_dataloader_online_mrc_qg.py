# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from operator import is_not
from functools import partial
import random
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from collections import OrderedDict

from utils.augment import insert_punctuation_marks
import random


"""
@Auth: Xhw
@Description: CHIP/CBLUE 医学实体关系抽取，数据来源 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from utils.seq2struct_dataloader_online import data_generator_single_schema
from utils.seq2struct_dataloader import (deleteDuplicate_v1, char_span_to_token_span, 
                                         token_span_to_char_span, get_token2char_char2token, 
                                         sequence_padding, search_all, search)

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from collections import namedtuple
_DocSpan = namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])

def slide_window(all_doc_tokens, max_length, doc_stride, offset=32):
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_length - offset:
            length = max_length - offset
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    return doc_spans

def load_mrc_schema(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            schema_dict = {
                'type':line,
                'role_list':[{'role':line, 'type':''}]
            }
            D.append(schema_dict)
    return D

def load_squad_style_data(filename):
    """See base class."""
    D = []
    data = json.load(open(filename, 'r', encoding='utf-8'))
    for content in data['data']:
        for para in content['paragraphs']:
            context = para['context']
            qas = para['qas']
            for qa in qas:
                question = qa['question']
                
                d = {
                    'text_a': context,
                    'text_b': question,
                    'target_list':[]
                }
                
                for anss in qa['answers']:
                    if isinstance(anss, list):
                        for ans in anss:
                            tmp_dict = {
                                'type':'答案',
                                'role_list': []
                            }
                            ans_text = ans.get('text', '')
                            tmp_dict['role_list'].append({
                                'type':'',
                                'argument':ans_text,
                                'argument_start_index':ans.get('answer_start', -1),
                                'role':'答案'
                            })
                            d['target_list'].append(tmp_dict)
                    else:
                        ans_text = anss.get('text', '')
                        tmp_dict = {
                                'type':'答案',
                                'role_list': []
                            }
                        tmp_dict['role_list'].append({
                                    'type':'',
                                    'argument':ans_text,
                                    'argument_start_index':anss.get('answer_start', -1),
                                    'role':'答案'
                        })
                        d['target_list'].append(tmp_dict)
                    
            D.append(d)
    return D

class data_generator_mrc(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0):
        
        self.data = data
        self.doc_stride = doc_stride
        self.offset = offset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_dict = task_dict
        self.sep_token = task_dict['sep_token']
        self.seg_token = task_dict['seg_token']
        self.group_token = task_dict['group_token']
        self.start_token = task_dict['start_token']
        self.end_token = task_dict['end_token']
        self.sentinel_token = task_dict['sentinel_token']
        self.sentinel_start_idx = task_dict['sentinel_start_idx']
        self.greedy_search = self.task_dict.get('greedy_search', False)
        self.mode = mode
        self.build_data = build_data
        self.add_neg = self.task_dict.get('add_neg', False)
        self.add_role_shuffle = self.task_dict.get('role_shuffle', False)
        self.schema_shuffle = self.task_dict.get('schema_shuffle', False)
        self.role_schema_order = self.task_dict.get('role_schema_order', False)
        self.remove_dup = self.task_dict.get('remove_dup', False)
        self.add_spefical_tokens = self.task_dict.get('add_spefical_tokens', True)
        self.search_mode = self.task_dict.get('search_mode', 'token_id')
        self.nce_type = self.task_dict.get('nce_type', True)
        
        if self.add_spefical_tokens:
            self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.seg_token, self.sep_token, 
                                                                                                self.group_token, self.start_token, self.end_token] })
        
        self.schema_dict = {}
        for schema_dict in schema:
            if schema_dict['type'] not in self.schema_dict:
                self.schema_dict[schema_dict['type']] = {
                        'role2sentinel':OrderedDict({}),
                        'sentinel2role':OrderedDict({}),
                        'role2type':OrderedDict({}),
                        'type2role':OrderedDict({}),
                        'role_index': 0
                }
            role_index = self.schema_dict[schema_dict['type']]['role_index']
            for _, role_dict in enumerate(schema_dict['role_list']):
                role_type = role_dict['type'] + role_dict['role']
                if role_type in self.schema_dict[schema_dict['type']]['role2sentinel']:
                    continue
                self.schema_dict[schema_dict['type']]['role2sentinel'][role_type] = self.sentinel_token.format(role_index+self.sentinel_start_idx)
                self.schema_dict[schema_dict['type']]['sentinel2role'][self.sentinel_token.format(role_index+self.sentinel_start_idx)] = role_type
                self.schema_dict[schema_dict['type']]['role2type'][role_dict['role']] = role_type
                self.schema_dict[schema_dict['type']]['type2role'][role_type] = role_dict['role']
                role_index += 1
            self.schema_dict[schema_dict['type']]['role_index'] = role_index
            
        import json
        logger.info("*** current schema: %s ***", json.dumps(self.schema_dict, ensure_ascii=False))
        
        self.features = []
        for item in self.data:
            total_target_dict = {}
            for target_dict in item['target_list']:
                target_type = target_dict['type']
                if target_type not in total_target_dict:
                    total_target_dict[target_type] = []
                if self.remove_dup:
                    target_dict['role_list'] = deleteDuplicate_v1(target_dict['role_list'])
                    target_dict['role_list'] = sorted(target_dict['role_list'], key=lambda item:item['argument'])
                total_target_dict[target_type].append(target_dict)
                
            for target_type in total_target_dict:
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
            
            input_ids = self.tokenizer(item['text_a'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
            
                for target_type in total_target_dict:
                    content = {}
                    for key in item:
                        if key in ['target_list']:
                            content[key] = total_target_dict[target_type]
                        else:
                            content[key] = item[key]
                    content['span_start'] = span_start
                    content['span_end'] = span_end
                    self.features.append(content)
                
                current_target_type = set(list(total_target_dict.keys()))
                total_target_type = set(list(self.schema_dict.keys()))
                
                left_target_type = list(total_target_type - current_target_type)
                import random
                random.shuffle(left_target_type)
                
                if len(left_target_type) >= 1 and self.nce_type:
                    neg_content = {}
                    for key in item:
                        if key in ['target_list']:
                            # event_dict = {
                            #     'type':target_type,
                            #     'role_list':[]
                            # }
                            neg_content[key] = []
                        else:
                            neg_content[key] = item[key]
                    neg_content['span_start'] = span_start
                    neg_content['span_end'] = span_end
                    neg_content['candidate_type'] = left_target_type
                    self.features.append(neg_content)

        import random
        random.shuffle(self.features)
        self.labels = [label] * len(self.features)
        self._task_id = label
                
    def __len__(self):
        return len(self.features)

    def encoder(self, item):
        text = item["text_a"]
        query = item["text_b"]
                
        if not item['target_list']:
            item['target_list'] = [{
                            'type':'答案',
                            'role_list':[]
                        }]
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
        encoder_instruction_text = self.tokenizer(instruction_text, return_offsets_mapping=True, max_length=self.max_len, truncation=True, add_special_tokens=False)
        instruction_input_ids = encoder_instruction_text["input_ids"]
        instruction_token_type_ids = encoder_instruction_text["token_type_ids"] #RoBERTa不需要NSP任务
        instruction_attention_mask = encoder_instruction_text["attention_mask"] #RoBERTa不需要NSP任务
        
        # encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            total_target_dict[target_type].append(target_dict['role_list'])
            
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        
        output_list = []
        
        for target_type in total_target_dict:
            
            schema_dict = self.schema_dict[target_type]
            if self.task_dict['add_schema_type']:
                schema_strings = target_type + self.sep_token
            else:
                schema_strings = ''
            key_list = list(schema_dict['role2sentinel'])
            if self.schema_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(key_list)
    
            for role in key_list:
                schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

            # add query
            schema_strings = query + self.sep_token + schema_strings + self.sep_token

            encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            schema_input_ids = encoder_schema_text["input_ids"]
            schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            schema_attention_mask = encoder_schema_text["attention_mask"]
            
            if self.add_role_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(total_target_dict[target_type])
            target_strings = ''
            for role_list in total_target_dict[target_type]:
                if self.role_schema_order:
                    role_index_list = []
                    for key_index, key in enumerate(key_list):
                        for role_index, role_dict in enumerate(role_list):
                            if key == role_dict['type'] + role_dict['role']:
                                role_index_list.append(role_index)
                else:
                    role_index_list = range(len(role_list))

                target_dict = OrderedDict({})
                for role_index in role_index_list:
                    role_dict = role_list[role_index]
                    argument_start_index = role_dict.get('argument_start_index', -1)
                    role_type = role_dict['type'] + role_dict['role']
                    argument = role_dict['argument']
                    if argument_start_index != -1:
                        start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                        start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                        # print(argument, start_t, end_t, span_start, span_end, '====')
                        if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                            if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                        else:
                            if self.greedy_search:
                                if self.search_mode == 'token_id':
                                    arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    sh = search(arguemnt_ids, span_input_ids)
                                elif self.search_mode == 'string':
                                    span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                                    span_text = text[span_text_pos[0]:span_text_pos[1]]
                                    sh = search(argument, span_text)
                                if sh != -1:
                                    # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                    if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                        target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                                    
                    else:
                        if self.search_mode == 'token_id':
                            arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(arguemnt_ids, span_input_ids)
                        elif self.search_mode == 'string':
                            span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                            span_text = text[span_text_pos[0]:span_text_pos[1]]
                            sh = search(argument, span_text)
                        if sh != -1:
                            # target_strings += argument + schema_dict['role2sentinel'][role_type]
                            if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''

                add_group_flag = False
                for key_set in target_dict:
                    target_strings += "".join(list(key_set))
                    add_group_flag = True
                # if self.task_dict['add_schema_type'] and add_group_flag:
                #     target_strings += self.seg_token + schema_dict['role2sentinel'][target_type] # add target-type
                if add_group_flag:
                    target_strings += self.group_token

            target_strings += self.end_token
            encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            target_input_ids = encoder_target_text["input_ids"]
            target_token_type_ids = encoder_target_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            target_attention_mask = encoder_target_text["attention_mask"]

            output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               span_input_ids, span_type_ids, span_attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask))
            
        return output_list
    
    def __getitem__(self, idx):
        return self.encoder(self.features[idx])[0]
    
    def get_labels(self):
        return self.labels
    
    def get_task_id(self):
        return self._task_id
    
    @staticmethod
    def collate_unilm(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            input_ids = instruction_input_ids + input_ids + schema_input_ids
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            
            input_ids += target_input_ids
            attention_mask += [1] * len(target_input_ids)
            token_type_ids += [1] * len(target_input_ids)
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids

    @staticmethod
    def collate_unilm_rl(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids, batch_schema_ids = [], [], [], [], []
        batch_input_ids = []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            query_input_ids = instruction_input_ids + input_ids + schema_input_ids[:-1]
            end_id = schema_input_ids[-1]
            attention_mask = [1] * len(query_input_ids)
            token_type_ids = [0] * len(query_input_ids)
            
            # batch_schema_ids.append(schema_input_ids)
            batch_token_ids.append(query_input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_target_ids.append(target_input_ids)
            # batch_input_ids.append(instruction_input_ids + input_ids)
        
        # batch_schema_ids = torch.tensor(sequence_padding(batch_schema_ids)).long()
        # batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_target_ids = torch.tensor(sequence_padding(batch_target_ids)).long()
        
        batch_size = batch_token_ids.shape[0]
        batch_token_ids = torch.cat([batch_token_ids, end_id*torch.ones((batch_size,1)).long()], dim=1)
        batch_mask_ids = torch.cat([batch_mask_ids, torch.ones((batch_size,1)).long()], dim=1)
        batch_token_type_ids = torch.cat([batch_token_type_ids, torch.zeros((batch_size,1)).long()], dim=1)
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids
    
    @staticmethod
    def collate_t5_v1(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids + schema_input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
    @staticmethod
    def collate_t5_v2(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = schema_input_ids + target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)

class data_generator_mrc_qg(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0):
        
        self.data = data
        self.doc_stride = doc_stride
        self.offset = offset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_dict = task_dict
        self.sep_token = task_dict['sep_token']
        self.seg_token = task_dict['seg_token']
        self.group_token = task_dict['group_token']
        self.start_token = task_dict['start_token']
        self.end_token = task_dict['end_token']
        self.sentinel_token = task_dict['sentinel_token']
        self.sentinel_start_idx = task_dict['sentinel_start_idx']
        self.greedy_search = self.task_dict.get('greedy_search', False)
        self.mode = mode
        self.build_data = build_data
        self.add_neg = self.task_dict.get('add_neg', False)
        self.add_role_shuffle = self.task_dict.get('role_shuffle', False)
        self.schema_shuffle = self.task_dict.get('schema_shuffle', False)
        self.role_schema_order = self.task_dict.get('role_schema_order', False)
        self.remove_dup = self.task_dict.get('remove_dup', False)
        self.add_spefical_tokens = self.task_dict.get('add_spefical_tokens', True)
        self.search_mode = self.task_dict.get('search_mode', 'token_id')
        self.nce_type = self.task_dict.get('nce_type', True)
        
        if self.add_spefical_tokens:
            self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.seg_token, self.sep_token, 
                                                                                                self.group_token, self.start_token, self.end_token] })
        
        self.schema_dict = {}
        role_index = 0
        for schema_dict in schema:
            if schema_dict['type'] not in self.schema_dict:
                self.schema_dict[schema_dict['type']] = {
                        'role2sentinel':OrderedDict({}),
                        'sentinel2role':OrderedDict({}),
                        'role2type':OrderedDict({}),
                        'type2role':OrderedDict({})
                }
            for _, role_dict in enumerate(schema_dict['role_list']):
                role_type = role_dict['type'] + role_dict['role']
                if role_type in self.schema_dict[schema_dict['type']]['role2sentinel']:
                    continue
                self.schema_dict[schema_dict['type']]['role2sentinel'][role_type] = self.sentinel_token.format(role_index+self.sentinel_start_idx)
                self.schema_dict[schema_dict['type']]['sentinel2role'][self.sentinel_token.format(role_index+self.sentinel_start_idx)] = role_type
                self.schema_dict[schema_dict['type']]['role2type'][role_dict['role']] = role_type
                self.schema_dict[schema_dict['type']]['type2role'][role_type] = role_dict['role']
                role_index += 1
            
        import json
        logger.info("*** current schema: %s ***", json.dumps(self.schema_dict, ensure_ascii=False))
        
        self.features = []
        for item in self.data:
            total_target_dict = {}
            for target_dict in item['target_list']:
                target_type = target_dict['type']
                if target_type not in total_target_dict:
                    total_target_dict[target_type] = []
                if self.remove_dup:
                    target_dict['role_list'] = deleteDuplicate_v1(target_dict['role_list'])
                    target_dict['role_list'] = sorted(target_dict['role_list'], key=lambda item:item['argument'])
                total_target_dict[target_type].append(target_dict)
                
            for target_type in total_target_dict:
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
            
            encoder_text = self.tokenizer(item['text_a'], return_offsets_mapping=True, add_special_tokens=False)
            text = item['text_a']
            input_ids = encoder_text['input_ids']
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            
            token2char = encoder_text.offset_mapping
            char2token = [None] * len(text)
            for i, ((start, end)) in enumerate(token2char):
                char2token[start:end] = [i] * (end - start)
            
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                span_input_ids = input_ids[span_start:span_end]
            
                add_flag = False
                for target_type in total_target_dict:
                    for target_dict in total_target_dict[target_type]:
                        for role_dict in target_dict['role_list']:
                            argument_start_index = role_dict.get('argument_start_index', -1)
                            role_type = role_dict['type'] + role_dict['role']
                            argument = role_dict['argument']
                            
                            if argument or argument_start_index != -1:
                                arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                sh = search(arguemnt_ids, span_input_ids)
                                if sh != -1:
                                    add_flag = True
                                    break
                        if add_flag:
                            break
                    if add_flag:
                        break
                if add_flag:
                    content = {}
                    for key in item:
                        if key in ['target_list']:
                            content[key] = total_target_dict[target_type]
                        else:
                            content[key] = item[key]
                    content['span_start'] = span_start
                    content['span_end'] = span_end
                    self.features.append(content)

        import random
        random.shuffle(self.features)
        self.labels = [label] * len(self.features)
        self._task_id = label
                
    def __len__(self):
        return len(self.features)
    
    def encoder(self, item):
        text = item["text_a"]
        query = item['text_b']
        
        if not item['target_list']:
            item['target_list'] = [{
                            'type':'答案',
                            'role_list':[]
                        }]
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
        encoder_instruction_text = self.tokenizer(instruction_text, return_offsets_mapping=True, max_length=self.max_len, truncation=True, add_special_tokens=False)
        instruction_input_ids = encoder_instruction_text["input_ids"]
        instruction_token_type_ids = encoder_instruction_text["token_type_ids"] #RoBERTa不需要NSP任务
        instruction_attention_mask = encoder_instruction_text["attention_mask"] #RoBERTa不需要NSP任务
        
        # encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)
            
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            total_target_dict[target_type].append(target_dict['role_list'])
            
        output_list = []

        target_strings = ""
        schema_strings = ""
        add_group_flag = False
        
        for target_type in total_target_dict:
            
            if self.task_dict['add_schema_type']:
                schema_strings += target_type + self.sep_token
            
            if self.add_role_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(total_target_dict[target_type])
            target_dict = OrderedDict({})
            for role_list in total_target_dict[target_type]:
                if self.role_schema_order:
                    role_index_list = []
                    for key_index, key in enumerate(key_list):
                        for role_index, role_dict in enumerate(role_list):
                            if key == role_dict['type'] + role_dict['role']:
                                role_index_list.append(role_index)
                else:
                    role_index_list = range(len(role_list))

                for role_index in role_index_list:
                    role_dict = role_list[role_index]
                    argument_start_index = role_dict.get('argument_start_index', -1)
                    role_type = role_dict['type'] + role_dict['role']
                    argument = role_dict['argument']
                    if argument_start_index != -1:
                        start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                        start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                        if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                            if (role_type, argument) not in target_dict:
                                target_dict[(role_type, argument)] = ''
                        else:
                            if self.greedy_search:
                                if self.search_mode == 'token_id':
                                    arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    sh = search(arguemnt_ids, span_input_ids)
                                elif self.search_mode == 'string':
                                    span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                                    span_text = text[span_text_pos[0]:span_text_pos[1]]
                                    sh = search(argument, span_text)
                                if sh != -1:
                                    # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                    if (role_type, argument) not in target_dict:
                                        target_dict[(role_type, argument)] = ''
                    else:
                        if self.search_mode == 'token_id':
                            arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(arguemnt_ids, span_input_ids)
                        elif self.search_mode == 'string':
                            span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                            span_text = text[span_text_pos[0]:span_text_pos[1]]
                            sh = search(argument, span_text)
                        if sh != -1:
                            # target_strings += argument + schema_dict['role2sentinel'][role_type]
                            if (role_type, argument) not in target_dict:
                                target_dict[(role_type, argument)] = ''
                                
                for key_set in target_dict:
                    schema_strings += ":".join(list(key_set)) + self.sep_token
                    add_group_flag = True
        
        if add_group_flag:
            target_strings += query + self.schema_dict['问题']['role2sentinel']['问题'] # add answer
        else:
            schema_strings += target_type + self.sep_token
        
        schema_strings += '问题' + self.schema_dict['问题']['role2sentinel']['问题'] + self.sep_token # add target-type
        encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        schema_input_ids = encoder_schema_text["input_ids"]
        schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        schema_attention_mask = encoder_schema_text["attention_mask"]

        target_strings += self.end_token
        encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        target_input_ids = encoder_target_text["input_ids"]
        target_token_type_ids = encoder_target_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        target_attention_mask = encoder_target_text["attention_mask"]

        output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
           span_input_ids, span_type_ids, span_attention_mask,
           schema_input_ids, schema_token_type_ids, schema_attention_mask,
           target_input_ids, target_token_type_ids, target_attention_mask))

        return output_list
    
    def __getitem__(self, idx):
        return self.encoder(self.features[idx])[0]
    
    def get_labels(self):
        return self.labels
    
    def get_task_id(self):
        return self._task_id
    
    @staticmethod
    def collate_unilm(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            input_ids = instruction_input_ids + input_ids + schema_input_ids
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            
            input_ids += target_input_ids
            attention_mask += [1] * len(target_input_ids)
            token_type_ids += [1] * len(target_input_ids)
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids
    
    @staticmethod
    def collate_unilm_rl(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids, batch_schema_ids = [], [], [], [], []
        batch_input_ids = []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            query_input_ids = instruction_input_ids + input_ids + schema_input_ids[:-1]
            end_id = schema_input_ids[-1]
            attention_mask = [1] * len(query_input_ids)
            token_type_ids = [0] * len(query_input_ids)
            
            # batch_schema_ids.append(schema_input_ids)
            batch_token_ids.append(query_input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_target_ids.append(target_input_ids)
            # batch_input_ids.append(instruction_input_ids + input_ids)
        
        # batch_schema_ids = torch.tensor(sequence_padding(batch_schema_ids)).long()
        # batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_target_ids = torch.tensor(sequence_padding(batch_target_ids)).long()
        
        batch_size = batch_token_ids.shape[0]
        batch_token_ids = torch.cat([batch_token_ids, end_id*torch.ones((batch_size,1)).long()], dim=1)
        batch_mask_ids = torch.cat([batch_mask_ids, torch.ones((batch_size,1)).long()], dim=1)
        batch_token_type_ids = torch.cat([batch_token_type_ids, torch.zeros((batch_size,1)).long()], dim=1)
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids
    
    
from utils.mask_utils import create_sentinel_ids, filter_input_ids, filter_target_ids, random_spans_noise_mask
class data_generator_unilm_pretrain(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0):
        self.data = data
        self.doc_stride = doc_stride
        self.offset = offset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_dict = task_dict
        self.sep_token = task_dict['sep_token']
        self.seg_token = task_dict['seg_token']
        self.group_token = task_dict['group_token']
        self.start_token = task_dict['start_token']
        self.end_token = task_dict['end_token']
        self.sentinel_token = task_dict['sentinel_token']
        self.sentinel_start_idx = task_dict['sentinel_start_idx']
        self.greedy_search = self.task_dict.get('greedy_search', False)
        self.mode = mode
        self.build_data = build_data
        self.add_neg = self.task_dict.get('add_neg', False)
        self.add_role_shuffle = self.task_dict.get('role_shuffle', False)
        self.schema_shuffle = self.task_dict.get('schema_shuffle', False)
        self.role_schema_order = self.task_dict.get('role_schema_order', False)
        self.remove_dup = self.task_dict.get('remove_dup', False)
        self.noise_density = self.task_dict.get('noise_density', 0.2)
        self.mean_noise_span_length = self.task_dict.get('mean_noise_span_length', 0.2)
        self.search_mode = self.task_dict.get('search_mode', 'token_id')
        
        self.add_spefical_tokens = self.task_dict.get('add_spefical_tokens', True)
        if self.add_spefical_tokens:
            self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.seg_token, self.sep_token, 
                                                                                                self.group_token, self.start_token, self.end_token] })
        
        from utils.mlm_generator import MLMGenerator
        self.mlm_generator = MLMGenerator(
                                self.task_dict.get('mask_ratio', 0.25), 
                                self.task_dict.get('random_ratio', 1e-10),
                                self.task_dict.get('min_tok', 2),
                                self.task_dict.get('max_tok', 10),
                                self.task_dict.get('mask_id', 103),
                                self.task_dict.get('pad', 0),
                                self.task_dict.get('geometric_p', 0.1),
                                self.tokenizer.get_vocab(),
                                self.task_dict.get('max_pair_targets', 72),
                                replacement_method='word_piece',
                                endpoints='')
        
        self.schema_dict = {}
        for schema_dict in schema:
            if schema_dict['type'] not in self.schema_dict:
                self.schema_dict[schema_dict['type']] = {
                        'role2sentinel':OrderedDict({}),
                        'sentinel2role':OrderedDict({}),
                        'role2type':OrderedDict({}),
                        'type2role':OrderedDict({}),
                        'role_index': 0
                }
            role_index = self.schema_dict[schema_dict['type']]['role_index']
            for _, role_dict in enumerate(schema_dict['role_list']):
                role_type = role_dict['type'] + role_dict['role']
                if role_type in self.schema_dict[schema_dict['type']]['role2sentinel']:
                    continue
                self.schema_dict[schema_dict['type']]['role2sentinel'][role_type] = self.sentinel_token.format(role_index+self.sentinel_start_idx)
                self.schema_dict[schema_dict['type']]['sentinel2role'][self.sentinel_token.format(role_index+self.sentinel_start_idx)] = role_type
                self.schema_dict[schema_dict['type']]['role2type'][role_dict['role']] = role_type
                self.schema_dict[schema_dict['type']]['type2role'][role_type] = role_dict['role']
                role_index += 1
            self.schema_dict[schema_dict['type']]['role_index'] = role_index
        
        self.features = []
        for item in self.data:
            input_ids = self.tokenizer(item['text_a'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                content = {}
                for key in item:
                    content[key] = item[key]
                content['span_start'] = span_start
                content['span_end'] = span_end
                self.features.append(content)
        
        import random
        random.shuffle(self.features)
        self.labels = [label] * len(self.features)
        self._task_id = label
                
    def __len__(self):
        return len(self.features)

    def encoder_unlabeled(self, item):
        text = item["text_a"]
        
        flag = False
        if text == "长城汽车上涨3% 上周四及周五获董事长增持\n客户端\n新浪港股讯，长城汽车(5.22,0.09,1.75%)（02333）H股现价升3.05%，报5.06元，盘中高见5.12元；成交约845万股，涉资4273万元。\nA股（沪：601633）现价8.1元人民币，升0.11元人民币，或升1.38%，成交1993万元人民币，涉及246万股．现价A股对H股呈溢价+74%。":
            print(item, '==========')
            flag = False
        
        if self.task_dict['instruction']:
            instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
        else:
            instruction_text = self.start_token
        encoder_instruction_text = self.tokenizer(instruction_text, return_offsets_mapping=True, max_length=self.max_len, truncation=True, add_special_tokens=False)
        instruction_input_ids = encoder_instruction_text["input_ids"]
        instruction_token_type_ids = encoder_instruction_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        instruction_attention_mask = encoder_instruction_text["attention_mask"] #RoBERTa不需要NSP任务
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end]
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        
        try:
            [masked_sent, 
              masked_target, 
              _] = self.mlm_generator.ner_span_mask(
                        span_input_ids, 
                        self.tokenizer,
                        entity_spans=None,
                        return_only_spans=False,
                        ner_masking_prob=0.2,
                        mask_num=self.task_dict.get('max_pair_targets', len(span_input_ids)*self.task_dict.get('mask_ratio', 0.25))
                       )
        except:
            print(span_input_ids)
        
        mask_indices = np.array([masked_target != 0]) # [ 0  1 -1  0  0  0  0  2 -1 -1  0  0  0  0  3 -1 -1 -1]
        input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8))[0]
        ilm_input_ids = []
        ilm_target_ids = []
        
#         for idx, mask_indice in enumerate(input_ids_sentinel):
#             if mask_indice == 0:
#                 ilm_input_ids += [masked_sent[idx]]
#             if mask_indice > 0:
#                 encoder_sentinel = self.tokenizer(self.sentinel_token.format(self.sentinel_start_idx+mask_indice), return_offsets_mapping=True, truncation=False, add_special_tokens=False)
#                 ilm_input_ids += encoder_sentinel['input_ids']
#                 ilm_target_ids += [masked_target[idx]]

#             if mask_indice < 0:
#                 ilm_target_ids += [masked_target[idx]]
#                 if idx < len(input_ids_sentinel) - 1:
#                     if input_ids_sentinel[idx+1] >= 0:
#                         ilm_target_ids += encoder_sentinel['input_ids']

#         if mask_indice < 0:
#             ilm_target_ids += encoder_sentinel['input_ids']

        if_add_sentinel = True
        for idx, mask_indice in enumerate(input_ids_sentinel):
            if mask_indice == 0:
                ilm_input_ids += [span_input_ids[idx]]
                if not if_add_sentinel:
                    ilm_target_ids += encoder_sentinel['input_ids']
                    if_add_sentinel = True
            if mask_indice > 0:
                if not if_add_sentinel:
                    ilm_target_ids += encoder_sentinel['input_ids']
                    if_add_sentinel = True
                
                encoder_sentinel = self.tokenizer(self.sentinel_token.format(self.sentinel_start_idx+mask_indice), return_offsets_mapping=True, truncation=False, add_special_tokens=False)
                ilm_input_ids += encoder_sentinel['input_ids']
                ilm_target_ids += [span_input_ids[idx]]
                if_add_sentinel = False

            if mask_indice < 0:
                ilm_target_ids += [span_input_ids[idx]]
                if_add_sentinel = False

        # if mask_indice < 0:
        if not if_add_sentinel:
            ilm_target_ids += encoder_sentinel['input_ids']
            
        ilm_input_ids += self.tokenizer(self.sep_token, return_offsets_mapping=True, truncation=False, add_special_tokens=False)['input_ids']
        ilm_attention_mask = [1] * len(ilm_input_ids)
        ilm_token_type_ids = [0] * len(ilm_input_ids)
        
        ilm_target_ids += self.tokenizer(self.end_token, return_offsets_mapping=True, truncation=False, add_special_tokens=False)['input_ids']
        ilm_target_attention_mask = [1] * len(ilm_input_ids)
        ilm_target_token_type_ids = [0] * len(ilm_input_ids)
            
        output_list = []
        output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               ilm_input_ids, ilm_token_type_ids, ilm_attention_mask,
               [], [], [],
               ilm_target_ids, ilm_target_token_type_ids, ilm_target_attention_mask))
            
        return output_list

    def encoder_labeled(self, item):
        text = item["text_a"]
        
        if self.task_dict['instruction']:
            instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
        else:
            instruction_text = self.start_token
        encoder_instruction_text = self.tokenizer(instruction_text, return_offsets_mapping=True, max_length=self.max_len, truncation=True, add_special_tokens=False)
        instruction_input_ids = encoder_instruction_text["input_ids"]
        instruction_token_type_ids = encoder_instruction_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        instruction_attention_mask = encoder_instruction_text["attention_mask"] #RoBERTa不需要NSP任务
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)
            
        span_start = item['span_start']
        span_end = item['span_end']
        
        span_input_ids = input_ids[span_start:span_end]
        
        entity_span = []
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            for role_dict in target_dict['role_list']:
                argument_start_index = role_dict.get('argument_start_index', -1)
                argument = role_dict['argument']
                if argument:
                    if self.search_mode == 'token_id':
                        arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                        sh = search_all(arguemnt_ids, span_input_ids)
                    elif self.search_mode == 'string':
                        span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                        span_text = text[span_text_pos[0]:span_text_pos[1]]
                        sh = search_all(argument, span_text)
                    if sh:
                        for sh_ in sh:
                            entity_span.append([sh_, sh_+len(arguemnt_ids)-1])
            
        output_list = []

        try:
            [masked_sent, 
              masked_target, 
              _] = self.mlm_generator.ner_span_mask(
                        span_input_ids, 
                        self.tokenizer,
                        entity_spans=entity_span,
                        return_only_spans=False,
                        ner_masking_prob=0.6,
                        mask_num=self.task_dict.get('max_pair_targets', len(span_input_ids)*self.task_dict.get('mask_ratio', 0.25))
                       )
        except:
            print(span_input_ids, '=====', entity_span)
                
        mask_indices = np.array([masked_target != 0]) # [ 0  1 -1  0  0  0  0  2 -1 -1  0  0  0  0  3 -1 -1 -1]
        input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8))[0]
        ilm_input_ids = []
        ilm_target_ids = []
        
#         for idx, mask_indice in enumerate(input_ids_sentinel):
#             if mask_indice == 0:
#                 ilm_input_ids += [masked_sent[idx]]
#             if mask_indice > 0:
#                 encoder_sentinel = self.tokenizer(self.sentinel_token.format(self.sentinel_start_idx+mask_indice), return_offsets_mapping=True, truncation=False, add_special_tokens=False)
#                 ilm_input_ids += encoder_sentinel['input_ids']
#                 ilm_target_ids += [masked_target[idx]]

#             if mask_indice < 0:
#                 ilm_target_ids += [masked_target[idx]]
#                 if idx < len(input_ids_sentinel) - 1:
#                     if input_ids_sentinel[idx+1] >= 0:
#                         ilm_target_ids += encoder_sentinel['input_ids']

#         if mask_indice < 0:
#             ilm_target_ids += encoder_sentinel['input_ids']

        if_add_sentinel = True
        for idx, mask_indice in enumerate(input_ids_sentinel):
            if mask_indice == 0:
                ilm_input_ids += [span_input_ids[idx]]
                if not if_add_sentinel:
                    ilm_target_ids += encoder_sentinel['input_ids']
                    if_add_sentinel = True
            if mask_indice > 0:
                if not if_add_sentinel:
                    ilm_target_ids += encoder_sentinel['input_ids']
                    if_add_sentinel = True
                
                encoder_sentinel = self.tokenizer(self.sentinel_token.format(self.sentinel_start_idx+mask_indice), return_offsets_mapping=True, truncation=False, add_special_tokens=False)
                ilm_input_ids += encoder_sentinel['input_ids']
                ilm_target_ids += [span_input_ids[idx]]
                if_add_sentinel = False

            if mask_indice < 0:
                ilm_target_ids += [span_input_ids[idx]]
                if_add_sentinel = False

        # if mask_indice < 0:
        if not if_add_sentinel:
            ilm_target_ids += encoder_sentinel['input_ids']
            
        ilm_input_ids += self.tokenizer(self.sep_token, return_offsets_mapping=True, truncation=False, add_special_tokens=False)['input_ids']
        ilm_attention_mask = [1] * len(ilm_input_ids)
        ilm_token_type_ids = [0] * len(ilm_input_ids)
        
        ilm_target_ids += self.tokenizer(self.end_token, return_offsets_mapping=True, truncation=False, add_special_tokens=False)['input_ids']
        ilm_target_attention_mask = [1] * len(ilm_input_ids)
        ilm_target_token_type_ids = [0] * len(ilm_input_ids)
        
        output_list = []
        output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               ilm_input_ids, ilm_token_type_ids, ilm_attention_mask,
               [], [], [],
               ilm_target_ids, ilm_target_token_type_ids, ilm_target_attention_mask))
            
        return output_list
    
    def __getitem__(self, idx):
        import random
        # return self.encoder_unlabeled(self.features[idx])[0]
        #return self.encoder_labeled(self.features[idx])[0]
        if random.random() > 0.5:
            return self.encoder_unlabeled(self.features[idx])[0]
        else:
            return self.encoder_labeled(self.features[idx])[0]
    
    def get_labels(self):
        return self.labels
    
    def get_task_id(self):
        return self._task_id

    @staticmethod
    def collate_unilm(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            input_ids = instruction_input_ids + input_ids + schema_input_ids
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            
            input_ids += target_input_ids
            attention_mask += [1] * len(target_input_ids)
            token_type_ids += [1] * len(target_input_ids)
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids
    
    @staticmethod
    def collate_unilm_rl(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids, batch_schema_ids = [], [], [], [], []
        batch_input_ids = []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            query_input_ids = instruction_input_ids + input_ids + schema_input_ids[:-1]
            end_id = schema_input_ids[-1]
            attention_mask = [1] * len(query_input_ids)
            token_type_ids = [0] * len(query_input_ids)
            
            # batch_schema_ids.append(schema_input_ids)
            batch_token_ids.append(query_input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_target_ids.append(target_input_ids)
            # batch_input_ids.append(instruction_input_ids + input_ids)
        
        # batch_schema_ids = torch.tensor(sequence_padding(batch_schema_ids)).long()
        # batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_target_ids = torch.tensor(sequence_padding(batch_target_ids)).long()
        
        batch_size = batch_token_ids.shape[0]
        batch_token_ids = torch.cat([batch_token_ids, end_id*torch.ones((batch_size,1)).long()], dim=1)
        batch_mask_ids = torch.cat([batch_mask_ids, torch.ones((batch_size,1)).long()], dim=1)
        batch_token_type_ids = torch.cat([batch_token_type_ids, torch.zeros((batch_size,1)).long()], dim=1)
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids
    
    @staticmethod
    def collate_t5_v1(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids + schema_input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(input_ids)
            batch_encoder_mask_ids.append(attention_mask)
            batch_encoder_token_type_ids.append(token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
    @staticmethod
    def collate_t5_v2(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = schema_input_ids + target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(input_ids)
            batch_encoder_mask_ids.append(attention_mask)
            batch_encoder_token_type_ids.append(token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)