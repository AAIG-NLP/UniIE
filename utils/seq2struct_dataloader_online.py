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

"""
@Auth: Xhw
@Description: CHIP/CBLUE 医学实体关系抽取，数据来源 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.augment import insert_punctuation_marks
import random
import logging

from utils.seq2struct_dataloader import (deleteDuplicate_v1, char_span_to_token_span, 
                                         token_span_to_char_span, get_token2char_char2token, 
                                         sequence_padding, search_all, search)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            D.append({
                "text":line["text"],
                "target_list":[]
            })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D

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

class data_generator_single_schema(Dataset):
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
            
            input_ids = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
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
        text = item["text"]
        
        import random
        if 'candidate_type' in item:
            random.shuffle(item['candidate_type'])
            event_dict = {
                            'type':item['candidate_type'][0],
                            'role_list':[]
                        }
            item['target_list'] = [event_dict]
        
        flag = False
        if '大连百傲化学股份有限公司关于股东股权解质押及再质押的公告' in text:
            print(item, '==========')
            flag = True
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
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
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            if self.remove_dup:
                role_list = deleteDuplicate_v1(target_dict['role_list'])
                role_list = sorted(role_list, key=lambda item:item['argument'])
            else:
                role_list = target_dict['role_list']
            total_target_dict[target_type].append(role_list)
            
        if flag:
            print(total_target_dict, '=========before========')
        
        for target_type in total_target_dict:
            before_num = len(total_target_dict[target_type])
            if before_num >= 2:
                # print(total_target_dict[target_type], '=========before========')
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                after_num = len(total_target_dict[target_type])
                # if before_num != after_num:
                #     print(total_target_dict[target_type], '=========after========')
            
        if flag:
            print(total_target_dict, '========after=========')
            
        if total_target_dict:
            assert len(total_target_dict) == 1
            
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

            schema_strings += self.sep_token
            encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            schema_input_ids = encoder_schema_text["input_ids"]
            schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            schema_attention_mask = encoder_schema_text["attention_mask"]

            target_strings = ''
            if self.add_role_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(total_target_dict[target_type])
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
                if flag:
                    print(target_dict, '=====target_dict====', target_strings)

                if add_group_flag:
                    target_strings += self.group_token
                    
                if flag:
                    print(target_dict, '=====target_dict====', target_strings)

            target_strings += self.end_token
            encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            target_input_ids = encoder_target_text["input_ids"]
            target_token_type_ids = encoder_target_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            target_attention_mask = encoder_target_text["attention_mask"]
            
            if self.task_dict.get('augment_marks', False):
                import random
                if random.random() > 0.4:
                    span_sentence = " ".join(self.tokenizer.tokenize(self.tokenizer.decode(span_input_ids)))
                    span_sentence = insert_punctuation_marks(span_sentence)
                    span_input_ids = self.tokenizer(span_sentence, add_special_tokens=False)['input_ids']
                    span_type_ids = [0] * len(span_input_ids)
                    span_attention_mask = [1] * len(span_input_ids)
                    
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
    
class data_generator_single_schema_cardinality(data_generator_single_schema):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0):
        
        super().__init__(data, tokenizer, max_len, schema, label, 
                 task_dict, mode, build_data, add_neg, add_role_shuffle,
                doc_stride, offset)
        
    def encoder(self, item):
        text = item["text"]
        
        flag = False
        if text == "长城汽车上涨3% 上周四及周五获董事长增持\n客户端\n新浪港股讯，长城汽车(5.22,0.09,1.75%)（02333）H股现价升3.05%，报5.06元，盘中高见5.12元；成交约845万股，涉资4273万元。\nA股（沪：601633）现价8.1元人民币，升0.11元人民币，或升1.38%，成交1993万元人民币，涉及246万股．现价A股对H股呈溢价+74%。":
            print(item, '==========')
            flag = False
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
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
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            if self.remove_dup:
                role_list = deleteDuplicate_v1(target_dict['role_list'])
                role_list = sorted(role_list, key=lambda item:item['argument'])
            else:
                role_list = target_dict['role_list']
            total_target_dict[target_type].append(role_list)
            
        if flag:
            print(total_target_dict, '=========before========')
        
        for target_type in total_target_dict:
            before_num = len(total_target_dict[target_type])
            if before_num >= 2:
                # print(total_target_dict[target_type], '=========before========')
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                after_num = len(total_target_dict[target_type])
                # if before_num != after_num:
                #     print(total_target_dict[target_type], '=========after========')
            
        if flag:
            print(total_target_dict, '========after=========')
            
        if total_target_dict:
            assert len(total_target_dict) == 1
            
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

            # for role in key_list:
            #     schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

            # schema_strings += self.sep_token
            # schema_strings += str(len(total_target_dict[target_type])) + self.sentinel_token.format(99) + self.sep_token # add cardinality
            # encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            # schema_input_ids = encoder_schema_text["input_ids"]
            # schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            # schema_attention_mask = encoder_schema_text["attention_mask"]
            
            group_count = 0

            target_strings = ''
            if self.add_role_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(total_target_dict[target_type])
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
                if flag:
                    print(target_dict, '=====target_dict====')

                if add_group_flag:
                    group_count += 1
                    target_strings += self.group_token
                    
            for role in key_list:
                schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

            schema_strings += self.sep_token
            import random
            if random.random() > 0.5:
                schema_strings += str(group_count) + self.sentinel_token.format(99) + self.sep_token # add cardinality
            else:
                schema_strings += str(20) + self.sentinel_token.format(99) + self.sep_token # add cardinality
            
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
    
    @staticmethod
    def collate_unilm(examples):
        return data_generator_single_schema.collate_unilm(examples)
    
    @staticmethod
    def collate_unilm_rl(examples):
        return data_generator_single_schema.collate_unilm_rl(examples)
    
    @staticmethod
    def collate_t5_v1(examples):
        return data_generator_single_schema.collate_t5_v1(examples)
    
    @staticmethod
    def collate_t5_v2(examples):
        return data_generator_single_schema.collate_t5_v2(examples)
    
class data_generator_single_schema_cardinality_predict(data_generator_single_schema):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0):
        
        super().__init__(data, tokenizer, max_len, schema, label, 
                 task_dict, mode, build_data, add_neg, add_role_shuffle,
                doc_stride, offset)
        
    def encoder(self, item):
        text = item["text"]
        
        flag = False
        if text == "长城汽车上涨3% 上周四及周五获董事长增持\n客户端\n新浪港股讯，长城汽车(5.22,0.09,1.75%)（02333）H股现价升3.05%，报5.06元，盘中高见5.12元；成交约845万股，涉资4273万元。\nA股（沪：601633）现价8.1元人民币，升0.11元人民币，或升1.38%，成交1993万元人民币，涉及246万股．现价A股对H股呈溢价+74%。":
            print(item, '==========')
            flag = False
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
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
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            if self.remove_dup:
                role_list = deleteDuplicate_v1(target_dict['role_list'])
                role_list = sorted(role_list, key=lambda item:item['argument'])
            else:
                role_list = target_dict['role_list']
            total_target_dict[target_type].append(role_list)
            
        if flag:
            print(total_target_dict, '=========before========')
        
        for target_type in total_target_dict:
            before_num = len(total_target_dict[target_type])
            if before_num >= 2:
                # print(total_target_dict[target_type], '=========before========')
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                after_num = len(total_target_dict[target_type])
                # if before_num != after_num:
                #     print(total_target_dict[target_type], '=========after========')
            
        if flag:
            print(total_target_dict, '========after=========')
            
        if total_target_dict:
            assert len(total_target_dict) == 1
            
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

            # for role in key_list:
            #     schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

            # schema_strings += self.sep_token
            # schema_strings += str(len(total_target_dict[target_type])) + self.sentinel_token.format(99) + self.sep_token # add cardinality
            # encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            # schema_input_ids = encoder_schema_text["input_ids"]
            # schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            # schema_attention_mask = encoder_schema_text["attention_mask"]
            
            group_count = 0

            target_strings = ''
            if self.add_role_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(total_target_dict[target_type])
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
                if flag:
                    print(target_dict, '=====target_dict====')

                if add_group_flag:
                    group_count += 1
                    target_strings += self.group_token
                    
            for role in key_list:
                schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

            schema_strings += self.sep_token
            encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            schema_input_ids = encoder_schema_text["input_ids"]
            schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            schema_attention_mask = encoder_schema_text["attention_mask"]

            target_strings = str(group_count) + self.sentinel_token.format(99) + self.seg_token + target_strings
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
    
    @staticmethod
    def collate_unilm(examples):
        return data_generator_single_schema.collate_unilm(examples)
    
    @staticmethod
    def collate_unilm_rl(examples):
        return data_generator_single_schema.collate_unilm_rl(examples)
    
    @staticmethod
    def collate_t5_v1(examples):
        return data_generator_single_schema.collate_t5_v1(examples)
    
    @staticmethod
    def collate_t5_v2(examples):
        return data_generator_single_schema.collate_t5_v2(examples)
    
    
class data_generator_element(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
               doc_stride=16, offset=0):
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
        self.span_pos_order = self.task_dict.get('span_pos_order', 'normal')
        self.search_mode = self.task_dict.get('search_mode', 'token_id')
        
        self.add_spefical_tokens = self.task_dict.get('add_spefical_tokens', True)
        if self.add_spefical_tokens:
        
            self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.seg_token, self.sep_token, 
                                                                                                self.group_token, self.start_token, self.end_token] })
        
        self.schema_dict = {
                        'role2sentinel':{},
                        'sentinel2role':{},
                        'role2type':{},
                        'type2role':{},
                        'role_type_mapping':{}
                }
        self.schema_dict['role2sentinel']['要素值'] = self.sentinel_token.format(self.sentinel_start_idx)
        self.schema_dict['sentinel2role'][self.sentinel_token.format(self.sentinel_start_idx)] = '要素值'
        self.schema_dict['role2type']['要素值'] = '要素值'
        self.schema_dict['type2role']['要素值'] = '要素值'
        
        self.features = []
        for item in self.data:
            input_ids = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
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

        print(self.features[0], '====')
                
    def __len__(self):
        return len(self.features)

    def encoder(self, item):
        text = item["text"]
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
        encoder_instruction_text = self.tokenizer(instruction_text, return_offsets_mapping=True, max_length=self.max_len, truncation=True, add_special_tokens=False)
        instruction_input_ids = encoder_instruction_text["input_ids"]
        instruction_token_type_ids = encoder_instruction_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        instruction_attention_mask = encoder_instruction_text["attention_mask"] #RoBERTa不需要NSP任务
        
        text = re.sub('[\s\t]+', self.seg_token, text)
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)
        
        total_target_dict = set()
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            for role_dict in target_dict['role_list']:
                total_target_dict.add(('要素值', role_dict['argument'], role_dict.get('argument_start_index', -1)))
            
        output_list = []
        
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        schema_dict = self.schema_dict

        schema_strings = ''
        key_list = list(schema_dict['role2sentinel'])
        if self.schema_shuffle:
            import random
            if random.random() > 0.5:
                random.shuffle(key_list)

        for role in key_list:
            schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

        schema_strings += self.sep_token
        encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        schema_input_ids = encoder_schema_text["input_ids"]
        schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        schema_attention_mask = encoder_schema_text["attention_mask"]

        target_strings = ''
        target_dict = OrderedDict({})
        if self.add_role_shuffle:
            import random
            if random.random() > 0.5:
                random.shuffle(list(total_target_dict))
        for role_tuple in total_target_dict:
            argument_start_index = role_tuple[-1]
            role_type = role_tuple[0]
            argument = role_tuple[1]
            if argument_start_index != -1:
                start, end = argument_start_index, argument_start_index + len(argument) - 1
                start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                    # target_strings += argument + schema_dict['role2sentinel'][role_type]
                    if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                        target_dict[(argument, schema_dict['role2sentinel'][role_type])] = start_t
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
                                target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh
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
                        target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh

        if self.span_pos_order == 'pos_order':
            sorted_keys = sorted(list(target_dict.keys()), key=lambda item: target_dict[item], reverse=False)
        elif self.span_pos_order == 'random':
            sorted_keys = list(target_dict.keys())
            import random
            if random.random() > 0.5:
                random.shuffle(sorted_keys)
        else:
            sorted_keys = list(target_dict.keys())
            import random
            if random.random() > 0.5:
                random.shuffle(sorted_keys)

        for key_set in sorted_keys:
            target_strings += "".join(list(key_set))

        target_strings += self.end_token
        encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        target_input_ids = encoder_target_text["input_ids"]
        target_token_type_ids = encoder_target_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        target_attention_mask = encoder_target_text["attention_mask"]

        # print('==data_generator_element===', self.task_dict['instruction'], '===', self.tokenizer.decode(span_input_ids), '==', schema_strings, '====', target_strings)

        # output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
        #    input_ids, token_type_ids, attention_mask,
        #    schema_input_ids, schema_token_type_ids, schema_attention_mask,
        #    target_input_ids, target_token_type_ids, target_attention_mask))

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
    
    
class data_generator_elemnt_group(Dataset):
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
        self.element_start_idx = self.task_dict['element_start_idx']
        self.remove_dup = self.task_dict.get('remove_dup', False)
        self.span_pos_order = self.task_dict.get('span_pos_order', 'normal')
        self.search_mode = self.task_dict.get('search_mode', 'token_id')
        
        print(self.span_pos_order)
        self.add_spefical_tokens = self.task_dict.get('add_spefical_tokens', True)
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
            
            input_ids = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
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
                
        import random
        random.shuffle(self.features)
        
        self.labels = [label] * len(self.features)
        self._task_id = label
                
    def __len__(self):
        return len(self.features)

    def encoder(self, item):
        text = item["text"]
        
        flag = False
        if text == "长城汽车上涨3% 上周四及周五获董事长增持\n客户端\n新浪港股讯，长城汽车(5.22,0.09,1.75%)（02333）H股现价升3.05%，报5.06元，盘中高见5.12元；成交约845万股，涉资4273万元。\nA股（沪：601633）现价8.1元人民币，升0.11元人民币，或升1.38%，成交1993万元人民币，涉及246万股．现价A股对H股呈溢价+74%。":
            print(item, '==========')
            flag = False
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
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
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            if self.remove_dup:
                role_list = deleteDuplicate_v1(target_dict['role_list'])
                role_list = sorted(role_list, key=lambda item:item['argument'])
            else:
                role_list = target_dict['role_list']
            total_target_dict[target_type].append(role_list)
            
        if flag:
            for target_type in total_target_dict:
                print(total_target_dictp[target_type], '======', len(total_target_dict[target_type]))
                
        for target_type in total_target_dict:
            if len(total_target_dict[target_type]) >= 2:
                # print(total_target_dict[target_type], '=========before========')
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                # print(total_target_dict[target_type], '=========after========')
        
        # for target_type in total_target_dict:
        #     total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
        
        if total_target_dict:
            assert len(total_target_dict) == 1
            
        output_list = []
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        element_schema_dict = {
                'element2sentinel':OrderedDict({}),
                'sentinel2element':OrderedDict({}),
                'element2pos':OrderedDict({})
        }
        target_dict_list = OrderedDict({})
        schema_dict_str = OrderedDict({})

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

            target_dict_list[target_type] = []
            schema_dict_str[target_type] = schema_strings

            if self.add_role_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(total_target_dict[target_type])

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
                    if argument:
                        if argument_start_index != -1:
                            start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                            start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                            if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = start_t
                                if argument not in element_schema_dict['element2pos']:
                                    element_schema_dict['element2pos'][argument] = start_t
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
                                            target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh
                                        if argument not in element_schema_dict['element2pos']:
                                            element_schema_dict['element2pos'][argument] = sh
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
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh
                                if argument not in element_schema_dict['element2pos']:
                                    element_schema_dict['element2pos'][argument] = sh
                target_dict_list[target_type].append(target_dict)

        element_list = sorted(list(element_schema_dict['element2pos']), key=lambda key:element_schema_dict['element2pos'][key], reverse=False)
        element_idx = 1
        for element in element_list:
            element_schema_dict['element2sentinel'][element] = self.sentinel_token.format(element_idx+self.element_start_idx)
            element_schema_dict['sentinel2element'][self.sentinel_token.format(element_idx+self.element_start_idx)] = element
            element_idx += 1

        for target_type in schema_dict_str:
            schema_strings = schema_dict_str[target_type]
            schema_strings += self.sep_token

            for element in element_list:
                schema_strings += element + element_schema_dict['element2sentinel'][element]
            schema_strings += self.sep_token
            encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
            schema_input_ids = encoder_schema_text["input_ids"]
            schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            schema_attention_mask = encoder_schema_text["attention_mask"]

            target_strings = ''
            for target_dict in target_dict_list[target_type]:
                group_flag = False
                for (argument, role) in target_dict:
                    target_strings += element_schema_dict['element2sentinel'][argument] + role + self.seg_token
                    group_flag = True
                if group_flag:
                    target_strings += self.group_token

            target_strings += self.end_token
            encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
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
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids,
                batch_encoder_token_type_ids, 
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

class data_generator_elemnt_and_group(Dataset):
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
        self.element_start_idx = self.task_dict['element_start_idx']
        self.remove_dup = self.task_dict.get('remove_dup', False)
        self.span_pos_order = self.task_dict.get('span_pos_order', 'normal')
        self.search_mode = self.task_dict.get('search_mode', 'token_id')
        
        print(self.span_pos_order)
        self.add_spefical_tokens = self.task_dict.get('add_spefical_tokens', True)
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
            
            input_ids = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
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
                
        import random
        random.shuffle(self.features)
        
        self.labels = [label] * len(self.features)
        self._task_id = label
                
    def __len__(self):
        return len(self.features)

    def encoder(self, item):
        text = item["text"]
        
        flag = False
        if text == "长城汽车上涨3% 上周四及周五获董事长增持\n客户端\n新浪港股讯，长城汽车(5.22,0.09,1.75%)（02333）H股现价升3.05%，报5.06元，盘中高见5.12元；成交约845万股，涉资4273万元。\nA股（沪：601633）现价8.1元人民币，升0.11元人民币，或升1.38%，成交1993万元人民币，涉及246万股．现价A股对H股呈溢价+74%。":
            print(item, '==========')
            flag = False
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
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
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            if self.remove_dup:
                role_list = deleteDuplicate_v1(target_dict['role_list'])
                role_list = sorted(role_list, key=lambda item:item['argument'])
            else:
                role_list = target_dict['role_list']
            total_target_dict[target_type].append(role_list)
            
        if flag:
            for target_type in total_target_dict:
                print(total_target_dictp[target_type], '======', len(total_target_dict[target_type]))
                
        for target_type in total_target_dict:
            if len(total_target_dict[target_type]) >= 2:
                # print(total_target_dict[target_type], '=========before========')
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                # print(total_target_dict[target_type], '=========after========')
        
        # for target_type in total_target_dict:
        #     total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
        
        if total_target_dict:
            assert len(total_target_dict) == 1
            
        output_list = []
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        element_schema_dict = {
                'element2sentinel':OrderedDict({}),
                'sentinel2element':OrderedDict({}),
                'element2pos':OrderedDict({})
        }
        target_dict_list = OrderedDict({})
        schema_dict_str = OrderedDict({})

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

            target_dict_list[target_type] = []
            schema_dict_str[target_type] = schema_strings

            if self.add_role_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(total_target_dict[target_type])

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
                    if argument:
                        if argument_start_index != -1:
                            start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                            start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                            if input_ids[start_t:end_t-1] and start_t >= span_start and end_t <= span_end:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = start_t
                                if argument not in element_schema_dict['element2pos']:
                                    element_schema_dict['element2pos'][argument] = start_t
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
                                            target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh
                                        if argument not in element_schema_dict['element2pos']:
                                            element_schema_dict['element2pos'][argument] = sh
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
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh
                                if argument not in element_schema_dict['element2pos']:
                                    element_schema_dict['element2pos'][argument] = sh
                target_dict_list[target_type].append(target_dict)

        element_list = sorted(list(element_schema_dict['element2pos']), key=lambda key:element_schema_dict['element2pos'][key], reverse=False)
        if self.span_pos_order == 'random':
            import random
            if random.random() > 0.5:
                random.shuffle(element_list)
        element_idx = 1
        for element in element_list:
            element_schema_dict['element2sentinel'][element] = self.sentinel_token.format(element_idx+self.element_start_idx)
            element_schema_dict['sentinel2element'][self.sentinel_token.format(element_idx+self.element_start_idx)] = element
            element_idx += 1
        
        for target_type in schema_dict_str:
            schema_strings = schema_dict_str[target_type]
            schema_strings += self.sep_token
            
            encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
            schema_input_ids = encoder_schema_text["input_ids"]
            schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            schema_attention_mask = encoder_schema_text["attention_mask"]

            target_strings = ''
            # first generate element
            for element in element_list:
                target_strings += element + element_schema_dict['element2sentinel'][element]
            target_strings += self.sentinel_token.format(99)
            
             # then generate group or type-detection
            for target_dict in target_dict_list[target_type]:
                group_flag = False
                for (argument, role) in target_dict:
                    target_strings += element_schema_dict['element2sentinel'][argument] + role + self.seg_token
                    group_flag = True
                if group_flag:
                    target_strings += self.group_token

            target_strings += self.end_token
            encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
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
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids,
                batch_encoder_token_type_ids, 
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
            input_ids = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
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
        text = item["text"]
        
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
        text = item["text"]
        
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
    
class data_generator_schema_type_generation(data_generator_single_schema):
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
        
        self.schema_dict = {
            'role2sentinel':{},
            'sentinel2role':{},
            'role2type':{},
            'type2role':{}
        
        }
        self.schema_dict['role2sentinel']['类型'] = self.sentinel_token.format(self.sentinel_start_idx)
        self.schema_dict['sentinel2role'][self.sentinel_token.format(self.sentinel_start_idx)] = '类型'
        self.schema_dict['role2type']['类型'] = '类型'
        self.schema_dict['type2role']['类型'] = '类型'
        
        self.features = []
        for item in self.data:
            input_ids = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
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
        
    def encoder(self, item):
        text = item["text"]
        
        flag = False
        if text == "长城汽车上涨3% 上周四及周五获董事长增持\n客户端\n新浪港股讯，长城汽车(5.22,0.09,1.75%)（02333）H股现价升3.05%，报5.06元，盘中高见5.12元；成交约845万股，涉资4273万元。\nA股（沪：601633）现价8.1元人民币，升0.11元人民币，或升1.38%，成交1993万元人民币，涉及246万股．现价A股对H股呈溢价+74%。":
            print(item, '==========')
            flag = False
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
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

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            if self.remove_dup:
                role_list = deleteDuplicate_v1(target_dict['role_list'])
                role_list = sorted(role_list, key=lambda item:item['argument'])
            else:
                role_list = target_dict['role_list']
            total_target_dict[target_type].append(role_list)
            
        total_target_set = []
        for target_type in total_target_dict:
            role_count = 0
            for role_list in total_target_dict[target_type]:
                for role_dict in role_list:
                    argument = role_dict['argument']
                    # arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                    # sh = search(arguemnt_ids, span_input_ids)
                    if self.search_mode == 'token_id':
                        arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                        sh = search(arguemnt_ids, span_input_ids)
                    elif self.search_mode == 'string':
                        span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                        span_text = text[span_text_pos[0]:span_text_pos[1]]
                        sh = search(argument, span_text)
                    if sh != -1:
                        role_count += 1
                        
            if role_count >= 2:
                total_target_set.append(target_type)
            
        schema_strings = '类型' + self.schema_dict['role2sentinel']['类型'] + self.sep_token
        import random
        if random.random() > 0.5:
            schema_strings += str(len(total_target_set)) + self.sentinel_token.format(99) + self.sep_token
        else:
            schema_strings += str(20) + self.sentinel_token.format(99) + self.sep_token
        encoder_schame_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        schema_input_ids = encoder_schame_text["input_ids"]
        schema_token_type_ids = encoder_schame_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        schema_attention_mask = encoder_schame_text["attention_mask"]
        
        import random
        target_strings = ""
        random.shuffle(list(total_target_set))
        for target_type in total_target_set:
            target_strings += target_type + self.schema_dict['role2sentinel']['类型'] + self.group_token
        
        target_strings += self.end_token
        encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        target_input_ids = encoder_target_text["input_ids"]
        target_token_type_ids = encoder_target_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        target_attention_mask = encoder_target_text["attention_mask"]

        output_list = []
        output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
           span_input_ids, span_type_ids, span_attention_mask,
           schema_input_ids, schema_token_type_ids, schema_attention_mask,
           target_input_ids, target_token_type_ids, target_attention_mask))
            
        return output_list 
    
    def __getitem__(self, idx):
        import random
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

class data_generator_flatten_schema(Dataset):
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
        
        if self.add_spefical_tokens:
            self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.seg_token, self.sep_token, 
                                                                                                self.group_token, self.start_token, self.end_token] })
        
        self.schema_dict = {
            'role2sentinel':{},
            'sentinel2role':{},
            'role2type':{},
            'type2role':{}
        }
        role_index = 0
        for schema_dict in schema:
            if schema_dict['type'] not in self.schema_dict['role2sentinel']:
                self.schema_dict['role2sentinel'][schema_dict['type']] = self.sentinel_token.format(role_index+self.sentinel_start_idx)
                self.schema_dict['sentinel2role'][self.sentinel_token.format(role_index+self.sentinel_start_idx)] = schema_dict['type']
                self.schema_dict['role2type'][schema_dict['type']] = schema_dict['type']
                self.schema_dict['type2role'][schema_dict['type']] = schema_dict['type']
                role_index += 1
                
        for schema_dict in schema:
            for _, role_dict in enumerate(schema_dict['role_list']):
                role_type = role_dict['type'] + role_dict['role']
                if role_type in self.schema_dict['role2sentinel']:
                    continue
                self.schema_dict['role2sentinel'][role_type] = self.sentinel_token.format(role_index+self.sentinel_start_idx)
                self.schema_dict['sentinel2role'][self.sentinel_token.format(role_index+self.sentinel_start_idx)] = role_type
                self.schema_dict['role2type'][role_dict['role']] = role_type
                self.schema_dict['type2role'][role_type] = role_dict['role']
                role_index += 1
                
        import json
        logger.info("*** current schema: %s ***", json.dumps(self.schema_dict, ensure_ascii=False))
        
        self.features = []
        for item in self.data:
            input_ids = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
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

    def encoder(self, item):
        text = item["text"]
        
        flag = False
        if text == "海钓比赛地点在厦门与金门之间的海域。":
            print(item, '==========')
            flag = True
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
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
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            if self.remove_dup:
                role_list = deleteDuplicate_v1(target_dict['role_list'])
                role_list = sorted(role_list, key=lambda item:item['argument'])
            else:
                role_list = target_dict['role_list']
            total_target_dict[target_type].append(role_list)
            
        if flag:
            print(total_target_dict, '=========before========')
        
        for target_type in total_target_dict:
            before_num = len(total_target_dict[target_type])
            if before_num >= 2:
                # print(total_target_dict[target_type], '=========before========')
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                after_num = len(total_target_dict[target_type])
                # if before_num != after_num:
                #     print(total_target_dict[target_type], '=========after========')
            
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        
        output_list = []
        
        schema_dict = self.schema_dict
        
        schema_strings = ''
        key_list = list(schema_dict['role2sentinel'])
        if self.schema_shuffle:
            import random
            if random.random() > 0.5:
                random.shuffle(key_list)

        for role in key_list:
            schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

        schema_strings += self.sep_token
        encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        schema_input_ids = encoder_schema_text["input_ids"]
        schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        schema_attention_mask = encoder_schema_text["attention_mask"]
        
        target_strings = ''
        for target_type in total_target_dict:
            if self.add_role_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(total_target_dict[target_type])
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
                if self.task_dict['add_schema_type'] and add_group_flag:
                    target_strings += self.seg_token + schema_dict['role2sentinel'][target_type] # add target-type
                if add_group_flag:
                    target_strings += self.group_token
                if flag:
                    print(target_dict, '=====target_dict====')

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
