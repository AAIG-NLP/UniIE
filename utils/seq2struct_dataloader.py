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
import random

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

def load_entity_schema(filename):
    """
    {"entity_type": "地理位置"}
    to 
    {"type": "裁员", "role_list": [{"role": "裁员方"}, {"role": "裁员人数"}, {"role": "时间"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            schema_dict = {
                'type':line['entity_type'],
                'role_list':[{'role':line['entity_type'], 'type':''}]
            }
            D.append(schema_dict)
    print(D)
    return D
    
def load_ie_schema(filename):
    """
    {"predicate": "父亲", "subject_type": "人物", "object_type": {"a1": "人物"}}
    to
    {"type": "裁员", "role_list": [{"role": "裁员方"}, {"role": "裁员人数"}, {"role": "时间"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            schema_dict = {
                'type':line['predicate'],
                'role_list':[{'role':line['subject_type'], 'type':'subject-'}]
            }
            for key in line['object_type']:
                schema_dict['role_list'].append({'role':line['object_type'][key], 'type':'object-'})
            D.append(schema_dict)
    return D

def load_ee_schema(filename):
    """
    {"predicate": "父亲", "subject_type": "人物", "object_type": {"a1": "人物"}}
    to
    {"type": "裁员", "role_list": [{"role": "裁员方"}, {"role": "裁员人数"}, {"role": "时间"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            schema_dict = {
                'type':line['event_type'],
                'role_list':[{'role':'触发词', 'type':''}]
            }
            for role_dict in line['role_list']:
                role_dict['type'] = ''
                schema_dict['role_list'].append(role_dict)
            D.append(schema_dict)
    return D

def load_entity(filename):
    """
    {"text": "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。", "entity_list": [{"start_idx": 3, "end_idx": 9, "type": "身体部位", "entity": "SARST细胞"}, {"start_idx": 19, "end_idx": 24, "type": "疾病", "entity": "成人SARS"}]}
    to
     {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            target_list = []
            entity_dict = {}
            for entity in line["entity_list"]:
                if entity['type'] not in entity_dict:
                    entity_dict[entity['type']] = []
                entity_dict[entity['type']].append(entity)
                
            # for entity_type in entity_dict:
            #     tmp_dict = {
            #         'type':entity_type,
            #         'role_list':[]
            #     }
            #     for entity in entity_dict[entity_type]:
            #         tmp_dict['role_list'].append({
            #             'role':entity_type,
            #             'argument':entity['entity'],
            #             'type':'',
            #             'argument_start_index':entity['start_idx']
            #         })
            #     target_list.append(tmp_dict)
            
            for entity_type in entity_dict:
                for entity in entity_dict[entity_type]:
                    tmp_dict = {
                        'type':entity_type,
                        'role_list':[]
                    }
                
                    tmp_dict['role_list'].append({
                        'role':entity_type,
                        'argument':entity['entity'],
                        'type':'',
                        'argument_start_index':entity['start_idx']
                    })
                    target_list.append(tmp_dict)
               
            D.append({
                "text":line["text"],
                "target_list":target_list
            })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
            
def load_duie(filename):
    """
    from {"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    to 
    {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            target_list = []
            for spo in line["spo_list"]:
                event_dict = {
                    'type':spo['predicate'],
                    'role_list':[]
                }
                event_dict['role_list'].append({'role':spo['subject_type'], 'argument':spo['subject'], 'type':'subject-', 'argument_start_index':-1})
                for key in spo['object_type']:
                    event_dict['role_list'].append({'role':spo['object_type'][key], 'argument':spo['object'][key], 'argument_start_index':-1, 'type':'object-'})
                target_list.append(event_dict)
            D.append({
                "text":line["text"],
                "target_list":target_list
            })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
    
def load_duee(filename):
    """
    {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            d = {'text': line['text'], 'target_list': []}
            for e in line["event_list"]:
                event_dict = {
                    'type':e['event_type'],
                    'role_list':[]
                }
                if e.get('trigger', None):
                    event_dict['role_list'].append(
                        {'role':'触发词', 'argument':e['trigger'], 'type':'', 'argument_start_index':e.get('trigger_start_index', -1)}
                    )
                for a in e['arguments']:
                     event_dict['role_list'].append((
                         {'role':a['role'], 'argument':a['argument'], 'type':'', 'argument_start_index':a.get('argument_start_index', -1)}
                    ))
                d['target_list'].append(event_dict)
            D.append(d)
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
    
import re
    
def load_entity_split(filename):
    """
    {"text": "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。", "entity_list": [{"start_idx": 3, "end_idx": 9, "type": "身体部位", "entity": "SARST细胞"}, {"start_idx": 19, "end_idx": 24, "type": "疾病", "entity": "成人SARS"}]}
    to
     {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            target_list = []
            entity_dict = {}
            for entity in line["entity_list"]:
                if entity['type'] not in entity_dict:
                    entity_dict[entity['type']] = []
                entity_dict[entity['type']].append(entity)
                
            # for entity_type in entity_dict:
            #     tmp_dict = {
            #         'type':entity_type,
            #         'role_list':[]
            #     }
            #     for entity in entity_dict[entity_type]:
            #         tmp_dict['role_list'].append({
            #             'role':entity_type,
            #             'argument':entity['entity'],
            #             'type':'',
            #             'argument_start_index':entity['start_idx']
            #         })
            #     target_list.append(tmp_dict)
            
            for entity_type in entity_dict:
                for entity in entity_dict[entity_type]:
                    tmp_dict = {
                        'type':entity_type,
                        'role_list':[]
                    }
                
                    tmp_dict['role_list'].append({
                        'role':entity_type,
                        'argument':entity['entity'],
                        'type':'',
                        'argument_start_index':entity['start_idx']
                    })
                    target_list.append(tmp_dict)
               
            D.append({
                "text":line["text"],
                "target_list":target_list
            })
            
            for text in [line['text']] + re.split('[\n。]', line['text']):
                D.append({
                    "text":text,
                    "target_list":target_list
                })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
            
def load_duie_split(filename):
    """
    from {"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    to 
    {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            target_list = []
            for spo in line["spo_list"]:
                event_dict = {
                    'type':spo['predicate'],
                    'role_list':[]
                }
                event_dict['role_list'].append({'role':spo['subject_type'], 'argument':spo['subject'], 'type':'subject-', 'argument_start_index':-1})
                for key in spo['object_type']:
                    event_dict['role_list'].append({'role':spo['object_type'][key], 'argument':spo['object'][key], 'argument_start_index':-1, 'type':'object-'})
                target_list.append(event_dict)
            D.append({
                "text":line["text"],
                "target_list":target_list
            })
            for text in [line['text']] + re.split('[\n。]', line['text']):
                D.append({
                    "text":text,
                    "target_list":target_list
                })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
    
def load_duee_split(filename):
    """
    {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            d = {'text': line['text'], 'target_list': []}
            for e in line["event_list"]:
                event_dict = {
                    'type':e['event_type'],
                    'role_list':[]
                }
                if e.get('trigger', None):
                    event_dict['role_list'].append(
                        {'role':'触发词', 'argument':e['trigger'], 'type':'', 'argument_start_index':e.get('trigger_start_index', -1)}
                    )
                for a in e['arguments']:
                     event_dict['role_list'].append((
                         {'role':a['role'], 'argument':a['argument'], 'type':'', 'argument_start_index':a.get('argument_start_index', -1)}
                    ))
                d['target_list'].append(event_dict)
            D.append(d)
            for text in [line['text']] + re.split('[\n。]', line['text']):
                D.append({
                    "text":text,
                    "target_list":d['target_list']
                })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
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
                                'start_index':ans.get('answer_start', -1),
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
                                    'start_index':anss.get('answer_start', -1),
                                    'role':'答案'
                        })
                        d['target_list'].append(tmp_dict)
                    
            D.append(d)
    return D
    
from functools import reduce
def deleteDuplicate_v1(input_dict_lst):
    f = lambda x,y:x if y in x else x + [y]
    return reduce(f, [[], ] + input_dict_lst)

def char_span_to_token_span(char2token, char_span):
    token_indexes = char2token[char_span[0]:char_span[1]]
    token_indexes = list(filter(partial(is_not, None), token_indexes))
    if token_indexes:
        return token_indexes[0], token_indexes[-1] + 1  # [start, end)
    else:  # empty
        return 0, 0

def token_span_to_char_span(token2char, token_span):
    char_indexes = token2char[token_span[0]:token_span[1]]
    char_indexes = [span for span in char_indexes]  # 删除CLS/SEP对应的span
    start, end = char_indexes[0][0], char_indexes[-1][1]
    return start, end

def get_token2char_char2token(tokenizer, text, maxlen):
    tokend = tokenizer(text, return_offsets_mapping=True, max_length=maxlen, truncation=True)
    token2char = tokend.offset_mapping
    
    char2token = [None] * len(text)
    for i, ((start, end)) in enumerate(token2char):
        char2token[start:end] = [i] * (end - start)
    
    return token2char, char2token

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def search_all(pattern, sequence):
    all_index = []
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            all_index.append(i)
    return all_index

def search_bin(bins, size):
    idx = len(bins) - 1
    for i, bin in enumerate(bins):
        if size <= bin:
            idx = i
            break
    return idx

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

class data_generator_single_schema_str(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False):
        self.data = data
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
        self.greedy_search = self.task_dict.get('greedy_search', True)
        self.mode = mode
        self.build_data = build_data
        self.add_neg = add_neg
        self.add_role_shuffle = add_role_shuffle
        self.schema_shuffle = self.task_dict.get('schema_shuffle', False)
        
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
                
        if self.build_data:
            self.features = []
            break_flag = False
            for item in self.data:
                for feature in self.encoder(item):
                    self.features.append(feature)
                    if self.mode == 'debug' and len(self.features) == 1000:
                        break_flag = True
                    if break_flag:
                        break
                if break_flag:
                    break

            self.labels = [label] * len(self.features)
            self._task_id = label
            
            print(self.features[0], '====')
                
    def __len__(self):
        return len(self.features)

    def encoder(self, item):
        text = item["text"]
        
        instruction_strings = self.task_dict['instruction'] + self.sep_token
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            total_target_dict[target_type].append(target_dict['role_list'])
            
        output_list = []
        
        doc_spans = slide_window(text, self.max_len, 64, offset=32)
        for doc_span in doc_spans:
            span_start = doc_span.start
            span_end = doc_span.start + doc_span.length
            
            span_strings = text[span_start:span_end] + self.sep_token
        
            for target_type in total_target_dict:
                schema_dict = self.schema_dict[target_type]
                if self.task_dict['add_schema_type']:
                    schema_strings = target_type + self.sep_token
                else:
                    schema_strings = ''
                
                key_list = list(schema_dict['role2sentinel'])
                if self.schema_shuffle:
                    random.shuffle(key_list)
                    
                for role in key_list:
                    schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

                # schema_strings += self.sep_token

                target_strings = ''
                if self.add_role_shuffle:
                    random.shuffle(total_target_dict[target_type])
                for role_list in total_target_dict[target_type]:
                    for role_dict in role_list:
                        argument_start_index = role_dict.get('argument_start_index', -1)
                        role_type = role_dict['type'] + role_dict['role']
                        argument = role_dict['argument']
                        sh = search(argument, span_strings)
                        if argument_start_index != -1:
                            if argument_start_index >= span_start and argument_start_index <= span_end - 1:
                                target_strings += argument + schema_dict['role2sentinel'][role_type]
                            else:
                                if self.greedy_search:
                                    if sh != -1:
                                        target_strings += argument + schema_dict['role2sentinel'][role_type]
                        else:
                            if sh != -1:
                                target_strings += argument + schema_dict['role2sentinel'][role_type]

                    if target_strings:
                        target_strings += self.group_token

                output_list.append((instruction_strings, span_strings, schema_strings, target_strings))
            
        return output_list
    
    def __getitem__(self, idx):
        return self.features[idx]

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
        
        if self.build_data:
            self.features = []
            break_flag = False
            for item in self.data:
                for feature in self.encoder(item):
                    self.features.append(feature)
                    if self.mode == 'debug' and len(self.features) == 1000:
                        break_flag = True
                    if break_flag:
                        break
                if break_flag:
                    break

            self.labels = [label] * len(self.features)
            self._task_id = label
            
            print(self.features[0], '====')
                
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
            
        output_list = []
        
        doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
        for doc_span in doc_spans:
            span_start = doc_span.start
            span_end = doc_span.start + doc_span.length
            
            span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
            
            span_type_ids = [0] * len(span_input_ids)
            span_attention_mask = [1] * len(span_input_ids)
        
            for target_type in total_target_dict:
                schema_dict = self.schema_dict[target_type]
                if self.task_dict['add_schema_type']:
                    schema_strings = target_type + self.sep_token
                else:
                    schema_strings = ''
                key_list = list(schema_dict['role2sentinel'])
                if self.schema_shuffle:
                    random.shuffle(key_list)
                    
                for role in key_list:
                    schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

                schema_strings += self.sep_token
                encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
                schema_input_ids = encoder_schema_text["input_ids"]
                schema_token_type_ids = encoder_schema_text["token_type_ids"] #RoBERTa不需要NSP任务
                schema_attention_mask = encoder_schema_text["attention_mask"]

                target_strings = ''
                if self.add_role_shuffle:
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
                            if input_ids[start_t:end_t] and start_t >= doc_span.start and end_t <= span_end:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                            else:
                                if self.greedy_search:
                                    arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    sh = search(arguemnt_ids, span_input_ids)
                                    if sh != -1:
                                        # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                        if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                            target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                        else:
                            arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(arguemnt_ids, span_input_ids)
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
                        target_strings += self.group_token

                target_strings += self.end_token
                encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
                target_input_ids = encoder_target_text["input_ids"]
                target_token_type_ids = encoder_target_text["token_type_ids"] #RoBERTa不需要NSP任务
                target_attention_mask = encoder_target_text["attention_mask"]

                # print(self.task_dict['instruction'], '===', text, '==', schema_strings, '====', target_strings )

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
        return self.features[idx]
    
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
            encdoer_attention_mask = [1] * len(input_ids)
            encoder_token_type_ids = [0] * len(input_ids)
            
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
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).long()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    

class data_generator_single_schema_single_role(Dataset):
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
        
        if self.build_data:
            self.features = []
            break_flag = False
            for item in self.data:
                for feature in self.encoder(item):
                    self.features.append(feature)
                    if self.mode == 'debug' and len(self.features) == 1000:
                        break_flag = True
                    if break_flag:
                        break
                if break_flag:
                    break

            self.labels = [label] * len(self.features)
            self._task_id = label
            
            print(self.features[0], '====')
                
    def __len__(self):
        return len(self.features)

    def encoder(self, item):
        text = item["text"]
        
        flag = False
        if text == "7月20日下午，在即墨区温泉街道四舍山滑翔基地发生了一起意外。教练于先生跟游客许女士一起体验滑翔伞飞行，在山顶飞出100多米后，从30多米的高空坠下。两人都被摔成重伤，而于教练受伤更重不幸去世。":
            print(item, '=====')
            flag = False
        
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
                total_target_dict[target_type] = {}
            for role_dict in target_dict['role_list']:
                role_type = role_dict['type'] + role_dict['role']
                if role_type not in total_target_dict[target_type]:
                    total_target_dict[target_type][role_type] = set()
                total_target_dict[target_type][role_type].add((role_type, role_dict['argument'], role_dict['argument_start_index']))
            
        if flag:
            print(total_target_dict)
        output_list = []
        
        doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
        for doc_span in doc_spans:
            span_start = doc_span.start
            span_end = doc_span.start + doc_span.length
            
            span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
            
            span_type_ids = [0] * len(span_input_ids)
            span_attention_mask = [1] * len(span_input_ids)
        
            for target_type in total_target_dict:
                schema_dict = self.schema_dict[target_type]
                
                for role_type in total_target_dict[target_type]:
                    role_tuple_list = list(total_target_dict[target_type][role_type])
                
                    if self.task_dict['add_schema_type']:
                        schema_strings = target_type + self.sep_token
                    else:
                        schema_strings = ''

                    schema_strings += role_type + schema_dict['role2sentinel'][role_type] # role sentinel

                    schema_strings += self.sep_token
                    encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
                    schema_input_ids = encoder_schema_text["input_ids"]
                    schema_token_type_ids = encoder_schema_text["token_type_ids"] #RoBERTa不需要NSP任务
                    schema_attention_mask = encoder_schema_text["attention_mask"]

                    target_strings = ''
                    target_dict = OrderedDict({})
                    
                    random.shuffle(role_tuple_list)

                    for role_tuple in role_tuple_list:
                        argument_start_index = role_tuple[-1]
                        role_type = role_tuple[0]
                        argument = role_tuple[1]
                        if argument_start_index != -1:
                            start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                            start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                            if flag:
                                print(role_tuple, start_t, end_t, doc_span.start, span_end, input_ids[start_t:end_t])
                            if input_ids[start_t:end_t] and start_t >= doc_span.start and end_t <= span_end:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                            else:
                                if self.greedy_search:
                                    arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    sh = search(arguemnt_ids, span_input_ids)
                                    if sh != -1:
                                        # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                        if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                            target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                        else:
                            arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(arguemnt_ids, span_input_ids)
                            if sh != -1:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                                    
                    if flag:
                        print(target_dict, '===')
                    for key_set in target_dict:
                        target_strings += "".join(list(key_set))

                    target_strings += self.end_token
                    encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
                    target_input_ids = encoder_target_text["input_ids"]
                    target_token_type_ids = encoder_target_text["token_type_ids"] #RoBERTa不需要NSP任务
                    target_attention_mask = encoder_target_text["attention_mask"]

                    # print('==data_generator_single_schema_single_role===', self.task_dict['instruction'], '===', self.tokenizer.decode(span_input_ids), '==', schema_strings, '====', target_strings)

                    # output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
                    #    input_ids, token_type_ids, attention_mask,
                    #    schema_input_ids, schema_token_type_ids, schema_attention_mask,
                    #    target_input_ids, target_token_type_ids, target_attention_mask))

                    output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
                       span_input_ids, span_type_ids, span_attention_mask,
                       schema_input_ids, schema_token_type_ids, schema_attention_mask,
                       target_input_ids, target_token_type_ids, target_attention_mask))
                    
                if self.add_neg:
                    exclude_role_type = set()
                    include_role_type = set()
                    for role_type in total_target_dict[target_type]:
                        include_role_type.add(role_type)
                    all_role_type = set()
                    for role_type in schema_dict['role2sentinel']:
                        all_role_type.add(role_type)
                    exclude_role_type = list(all_role_type - include_role_type)
                    
                    random.shuffle(exclude_role_type)
                    
                    for role_type in exclude_role_type[:2]:
                        if self.task_dict['add_schema_type']:
                            schema_strings = target_type + self.sep_token
                        else:
                            schema_strings = ''

                        schema_strings += role_type + schema_dict['role2sentinel'][role_type] # role sentinel

                        schema_strings += self.sep_token
                        encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
                        schema_input_ids = encoder_schema_text["input_ids"]
                        schema_token_type_ids = encoder_schema_text["token_type_ids"] #RoBERTa不需要NSP任务
                        schema_attention_mask = encoder_schema_text["attention_mask"]

                        target_strings = ''
                        target_strings += self.end_token
                        encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
                        target_input_ids = encoder_target_text["input_ids"]
                        target_token_type_ids = encoder_target_text["token_type_ids"] #RoBERTa不需要NSP任务
                        target_attention_mask = encoder_target_text["attention_mask"]
                        
                        # print('==negative data_generator_single_schema_single_role===', self.task_dict['instruction'], '===', self.tokenizer.decode(span_input_ids), '==', schema_strings, '====', target_strings)
                        
                        output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
                                       span_input_ids, span_type_ids, span_attention_mask,
                                       schema_input_ids, schema_token_type_ids, schema_attention_mask,
                                       target_input_ids, target_token_type_ids, target_attention_mask))
                        
            
        return output_list
    
    def __getitem__(self, idx):
        return self.features[idx]
    
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
            encdoer_attention_mask = [1] * len(input_ids)
            encoder_token_type_ids = [0] * len(input_ids)
            
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
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).long()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
class data_generator_element(Dataset):
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
        self.span_pos_order = self.task_dict.get('span_pos_order', 'normal')
        
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
        
        if self.build_data:
            self.features = []
            break_flag = False
            for item in self.data:
                for feature in self.encoder(item):
                    self.features.append(feature)
                    if self.mode == 'debug' and len(self.features) == 1000:
                        break_flag = True
                    if break_flag:
                        break
                if break_flag:
                    break

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
        
        total_target_dict = set()
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            for role_dict in target_dict['role_list']:
                total_target_dict.add(('要素值', role_dict['argument'], role_dict.get('argument_start_index', -1)))
            
        output_list = []
        
        doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
        for doc_span in doc_spans:
            span_start = doc_span.start
            span_end = doc_span.start + doc_span.length
            
            span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
            
            span_type_ids = [0] * len(span_input_ids)
            span_attention_mask = [1] * len(span_input_ids)
        
            schema_dict = self.schema_dict

            schema_strings = ''
            key_list = list(schema_dict['role2sentinel'])
            if self.schema_shuffle:
                random.shuffle(key_list)

            for role in key_list:
                schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

            schema_strings += self.sep_token
            encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            schema_input_ids = encoder_schema_text["input_ids"]
            schema_token_type_ids = encoder_schema_text["token_type_ids"] #RoBERTa不需要NSP任务
            schema_attention_mask = encoder_schema_text["attention_mask"]

            target_strings = ''
            target_dict = OrderedDict({})
            if self.add_role_shuffle:
                random.shuffle(total_target_dict)
            for role_tuple in total_target_dict:
                argument_start_index = role_tuple[-1]
                role_type = role_tuple[0]
                argument = role_tuple[1]
                if argument_start_index != -1:
                    start, end = argument_start_index, argument_start_index + len(argument) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t] and start_t >= doc_span.start and end_t <= span_end:
                        # target_strings += argument + schema_dict['role2sentinel'][role_type]
                        if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                            target_dict[(argument, schema_dict['role2sentinel'][role_type])] = start_t
                    else:
                        if self.greedy_search:
                            arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(arguemnt_ids, span_input_ids)
                            if sh != -1:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh
                else:
                    arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                    sh = search(arguemnt_ids, span_input_ids)
                    if sh != -1:
                        # target_strings += argument + schema_dict['role2sentinel'][role_type]
                        if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                            target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh

           
            if self.span_pos_order == 'pos_order':
                sorted_keys = sorted(list(target_dict.keys()), key=lambda item: target_dict[item], reverse=False)
            elif self.span_pos_order == 'random':
                sorted_keys = list(target_dict.keys())
                import random
                random.shuffle(sorted_keys)
            
            for key_set in sorted_keys:
                target_strings += "".join(list(key_set))
            
            target_strings += self.end_token
            encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            target_input_ids = encoder_target_text["input_ids"]
            target_token_type_ids = encoder_target_text["token_type_ids"] #RoBERTa不需要NSP任务
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
        return self.features[idx]
    
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
            encdoer_attention_mask = [1] * len(input_ids)
            encoder_token_type_ids = [0] * len(input_ids)
            
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
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).long()#RoBERTa 不需要NSP
        
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
        
        print(self.span_pos_order)
        
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
                
        if self.build_data:
            self.features = []
            break_flag = False
            for item in self.data:
                for feature in self.encoder(item):
                    self.features.append(feature)
                    if self.mode == 'debug' and len(self.features) == 1000:
                        break_flag = True
                    if break_flag:
                        break
                if break_flag:
                    break

            self.labels = [label] * len(self.features)
            self._task_id = label
            
            print(self.features[0], '====')
                
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
            
        output_list = []
        
        doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
        for doc_span in doc_spans:
            span_start = doc_span.start
            span_end = doc_span.start + doc_span.length
            
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
                    random.shuffle(key_list)
                    
                for role in key_list:
                    schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

                target_dict_list[target_type] = []
                schema_dict_str[target_type] = schema_strings
                
                if self.add_role_shuffle:
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
                            if input_ids[start_t:end_t] and start_t >= doc_span.start and end_t <= span_end:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = start_t
                                if argument not in element_schema_dict['element2pos']:
                                    element_schema_dict['element2pos'][argument] = start_t
                            else:
                                if self.greedy_search:
                                    arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    sh = search(arguemnt_ids, span_input_ids)
                                    if sh != -1:
                                        # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                        if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                            target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh
                                        if argument not in element_schema_dict['element2pos']:
                                            element_schema_dict['element2pos'][argument] = sh
                        else:
                            arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(arguemnt_ids, span_input_ids)
                            if sh != -1:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh
                                if argument not in element_schema_dict['element2pos']:
                                    element_schema_dict['element2pos'][argument] = sh
                    target_dict_list[target_type].append(target_dict)
                        
            if self.span_pos_order == 'pos_order':
                element_list = sorted(list(element_schema_dict['element2pos']), key=lambda key:element_schema_dict['element2pos'][key], reverse=False)
            elif self.span_pos_order == 'random':
                element_list = list(element_schema_dict['element2pos'])
                random.shuffle(element_list)

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
                schema_token_type_ids = encoder_schema_text["token_type_ids"] #RoBERTa不需要NSP任务
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
                target_token_type_ids = encoder_target_text["token_type_ids"] #RoBERTa不需要NSP任务
                target_attention_mask = encoder_target_text["attention_mask"]

                # print(self.task_dict['instruction'], '===', text, '==', schema_strings, '====', target_strings )

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
        return self.features[idx]
    
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
            encdoer_attention_mask = [1] * len(input_ids)
            encoder_token_type_ids = [0] * len(input_ids)
            
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
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).long()#RoBERTa 不需要NSP
        
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
        
        print(self.span_pos_order)
        
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
                
        if self.build_data:
            self.features = []
            break_flag = False
            for item in self.data:
                for feature in self.encoder(item):
                    self.features.append(feature)
                    if self.mode == 'debug' and len(self.features) == 1000:
                        break_flag = True
                    if break_flag:
                        break
                if break_flag:
                    break

            self.labels = [label] * len(self.features)
            self._task_id = label
            
            print(self.features[0], '====')
                
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
            
        output_list = []
        
        doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
        for doc_span in doc_spans:
            span_start = doc_span.start
            span_end = doc_span.start + doc_span.length
            
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
                    random.shuffle(key_list)
                    
                for role in key_list:
                    schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

                target_dict_list[target_type] = []
                schema_dict_str[target_type] = schema_strings
                
                if self.add_role_shuffle:
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
                            if input_ids[start_t:end_t] and start_t >= doc_span.start and end_t <= span_end:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = start_t
                                if argument not in element_schema_dict['element2pos']:
                                    element_schema_dict['element2pos'][argument] = start_t
                            else:
                                if self.greedy_search:
                                    arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    sh = search(arguemnt_ids, span_input_ids)
                                    if sh != -1:
                                        # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                        if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                            target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh
                                        if argument not in element_schema_dict['element2pos']:
                                            element_schema_dict['element2pos'][argument] = sh
                        else:
                            arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(arguemnt_ids, span_input_ids)
                            if sh != -1:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = sh
                                if argument not in element_schema_dict['element2pos']:
                                    element_schema_dict['element2pos'][argument] = sh
                    target_dict_list[target_type].append(target_dict)
                        
            if self.span_pos_order == 'pos_order':
                element_list = sorted(list(element_schema_dict['element2pos']), key=lambda key:element_schema_dict['element2pos'][key], reverse=False)
            elif self.span_pos_order == 'random':
                element_list = list(element_schema_dict['element2pos'])
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
                schema_token_type_ids = encoder_schema_text["token_type_ids"] #RoBERTa不需要NSP任务
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
                target_token_type_ids = encoder_target_text["token_type_ids"] #RoBERTa不需要NSP任务
                target_attention_mask = encoder_target_text["attention_mask"]

                # print(self.task_dict['instruction'], '===', text, '==', schema_strings, '====', target_strings )

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
        return self.features[idx]
    
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
            encdoer_attention_mask = [1] * len(input_ids)
            encoder_token_type_ids = [0] * len(input_ids)
            
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
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).long()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
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
        self.add_role_shuffle = add_role_shuffle
        self.schema_shuffle = self.task_dict.get('schema_shuffle', False)
        
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
                
        # print(self.schema_dict, '==schema_dict==')
        
        if self.build_data:
            self.features = []
            break_flag = False
            for item in self.data:
                for feature in self.encoder(item):
                    self.features.append(feature)
                    if self.mode == 'debug' and len(self.features) == 1000:
                        break_flag = True
                    if break_flag:
                        break
                if break_flag:
                    break

            self.labels = [label] * len(self.features)
            self._task_id = label
            
            print(self.features[0], '====')
                
    def __len__(self):
        return len(self.features)

    def encoder(self, item):
        text = item["text_a"]
        query = item["text_b"]
        
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
            
        output_list = []
        
        doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
        for doc_span in doc_spans:
            span_start = doc_span.start
            span_end = doc_span.start + doc_span.length
            
            span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
            
            span_type_ids = [0] * len(span_input_ids)
            span_attention_mask = [1] * len(span_input_ids)
            
            for target_type in total_target_dict:
                schema_dict = self.schema_dict[target_type]
                if self.task_dict['add_schema_type']:
                    schema_strings = target_type + self.sep_token
                else:
                    schema_strings = ''
                key_list = list(schema_dict['role2sentinel'])
                if self.schema_shuffle:
                    random.shuffle(key_list)
                    
                for role in key_list:
                    schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

                schema_strings = query + self.sep_token + schema_strings + self.sep_token

                encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
                schema_input_ids = encoder_schema_text["input_ids"]
                schema_token_type_ids = encoder_schema_text["token_type_ids"] #RoBERTa不需要NSP任务
                schema_attention_mask = encoder_schema_text["attention_mask"]

                target_strings = ''
                for role_list in total_target_dict[target_type]:
                    for role_dict in role_list:
                        argument_start_index = role_dict.get('argument_start_index', -1)
                        role_type = role_dict['type'] + role_dict['role']
                        argument = role_dict['argument']
                        if argument_start_index != -1:
                            start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                            start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                            if input_ids[start_t:end_t] and start_t >= doc_span.start and end_t <= span_end:
                                target_strings += argument + schema_dict['role2sentinel'][role_type]
                            else:
                                if self.greedy_search:
                                    arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    sh = search(arguemnt_ids, span_input_ids)
                                    if sh != -1:
                                        target_strings += argument + schema_dict['role2sentinel'][role_type]
                        else:
                            arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(arguemnt_ids, span_input_ids)
                            if sh != -1:
                                target_strings += argument + schema_dict['role2sentinel'][role_type]

                    if target_strings:
                        target_strings += self.group_token

                target_strings += self.end_token
                encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
                target_input_ids = encoder_target_text["input_ids"]
                target_token_type_ids = encoder_target_text["token_type_ids"] #RoBERTa不需要NSP任务
                target_attention_mask = encoder_target_text["attention_mask"]

                # print(self.task_dict['instruction'], '===', text, '==', schema_strings, '====', target_strings )

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
        return self.features[idx]
    
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
    def collate_t5_v1(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids + schema_input_ids
            encdoer_attention_mask = [1] * len(input_ids)
            encoder_token_type_ids = [0] * len(input_ids)
            
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
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).long()#RoBERTa 不需要NSP
        
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
        self.add_role_shuffle = add_role_shuffle
        self.add_neg = self.task_dict.get('add_neg', False)
        
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
        
        if self.build_data:
            self.features = []
            break_flag = False
            for item in self.data:
                for feature in self.encoder(item):
                    self.features.append(feature)
                    if self.mode == 'debug' and len(self.features) == 1000:
                        break_flag = True
                    if break_flag:
                        break
                if break_flag:
                    break
                    
            print(self.features[0], '====')
                
            self.labels = [label] * len(self.features)
            self._task_id = label
                
    def __len__(self):
        return len(self.features)
    
    def encoder(self, item):
        text = item["text_a"]
        query = item['text_b']
        
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
            
        output_list = []
        
        doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
        for doc_span in doc_spans:
            span_start = doc_span.start
            span_end = doc_span.start + doc_span.length
            
            span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
            
            span_type_ids = [0] * len(span_input_ids)
            span_attention_mask = [1] * len(span_input_ids)
        
            for target_dict in item['target_list']:
                schema_strings = ""
                valid_ans_flag = False
                for role_dict in target_dict['role_list']:
                    argument_start_index = role_dict.get('argument_start_index', -1)
                    role_type = role_dict['type'] + role_dict['role']
                    argument = role_dict['argument']
                    if argument_start_index != -1:
                        start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                        start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                        if input_ids[start_t:end_t] and start_t >= doc_span.start and end_t <= span_end:
                            schema_strings += role_type + self.sep_token + argument + self.sep_token
                            valid_ans_flag = True
                        else:
                            if self.greedy_search:
                                arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                sh = search(arguemnt_ids, span_input_ids)
                                if sh != -1:
                                    schema_strings += role_type + self.sep_token + argument + self.sep_token
                                    valid_ans_flag = True
                    else:
                        arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                        sh = search(arguemnt_ids, span_input_ids)
                        if sh != -1:
                            schema_strings += role_type + self.sep_token + argument + self.sep_token
                            valid_ans_flag = True

                if valid_ans_flag:
                    schema_strings += '问题' + self.schema_dict['问题']['role2sentinel']['问题'] + self.sep_token
                    target_strings = query + self.schema_dict['问题']['role2sentinel']['问题'] + self.end_token

                    encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
                    schema_input_ids = encoder_schema_text["input_ids"]
                    schema_token_type_ids = encoder_schema_text["token_type_ids"] #RoBERTa不需要NSP任务
                    schema_attention_mask = encoder_schema_text["attention_mask"]

                    encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
                    target_input_ids = encoder_target_text["input_ids"]
                    target_token_type_ids = encoder_target_text["token_type_ids"] #RoBERTa不需要NSP任务
                    target_attention_mask = encoder_target_text["attention_mask"]

                    # print(query, '===', text, '==', schema_strings, '====', target_strings, item['target_list'])

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
        return self.features[idx]
    
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
    def collate_t5_v1(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids + schema_input_ids
            encdoer_attention_mask = [1] * len(input_ids)
            encoder_token_type_ids = [0] * len(input_ids)
            
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
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).long()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
class data_generator_schema_cls(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True,
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
        self.mode = mode
        self.build_data = build_data
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.seg_token, self.sep_token, 
                                                                                                self.group_token, self.start_token, self.end_token] })
        
        self.schema_dict = {}
        self.schema2id = {}
        self.id2schema = {}
        schema_idx = 0
        
        for schema_dict in schema:
            if schema_dict['type'] not in self.schema_dict:
                self.schema2id[schema_dict['type']] = schema_idx
                self.id2schema[schema_idx] = schema_dict['type']
                schema_idx += 1
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
                
        # print(self.id2schema, '====')
        
        if self.build_data:
            self.features = []
            break_flag = False
            for item in self.data:
                for feature in self.encoder(item):
                    self.features.append(feature)
                    if self.mode == 'debug' and len(self.features) == 1000:
                        break_flag = True
                    if break_flag:
                        break
                if break_flag:
                    break
                    
            print(self.features[0], '====')
                
            self.labels = [label] * len(self.features)
            self._task_id = label
                
    def __len__(self):
        return len(self.features)
    
    def encoder(self, item):
        text = item["text"]
        
        # encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        
        output_list = []
        
        doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
        for doc_span in doc_spans:
            span_start = doc_span.start
            span_end = doc_span.start + doc_span.length
            
            span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] +input_ids[span_start:span_end] + self.tokenizer(self.end_token, add_special_tokens=False)['input_ids']
            
            span_type_ids = [0] * len(span_input_ids)
            span_attention_mask = [1] * len(span_input_ids)
            
            target_set = set()
            for target_dict in item['target_list']:
                target_set.add(target_dict['type'])
            
            labels = [0]*len(self.schema2id)
            for target_type in target_set:
                target_type_ids = self.schema2id[target_type]
                labels[target_type_ids] = 1
            
            output_list.append((text, 
               span_input_ids, span_type_ids, span_attention_mask,
               labels))
        return output_list
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def get_labels(self):
        return self.labels
    
    def get_task_id(self):
        return self._task_id

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_labels = []
        for item in examples:
            (text, input_ids, token_type_ids, attention_mask, labels) = item
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(labels)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_labels = torch.tensor(batch_labels).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_labels
    
class data_generator_schema_cls_v1(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True,
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
        self.mode = mode
        self.build_data = build_data
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.seg_token, self.sep_token, 
                                                                                                self.group_token, self.start_token, self.end_token] })
        
        self.schema_dict = {}
        self.schema2id = {}
        self.id2schema = {}
        schema_idx = 0
        
        for schema_dict in schema:
            if schema_dict['type'] not in self.schema_dict:
                self.schema2id[schema_dict['type']] = schema_idx
                self.id2schema[schema_idx] = schema_dict['type']
                schema_idx += 1
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
                
        # print(self.id2schema, '====')
        self.labels = [label] * len(self.data)
        self._task_id = label
                
    def __len__(self):
        return len(self.data)
    
    def encoder(self, item):
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            total_target_dict[target_type].append(target_dict['role_list'])
            
        target_set = set()
        for target_type in total_target_dict:
            target_type_cnt = 0
            for role_list in total_target_dict[target_type]:
                for role_dict in role_list:
                    argument = role_dict['argument']
                    arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                    sh = search(arguemnt_ids, input_ids) 
                    if sh != -1:
                        target_type_cnt += 1
            if target_type_cnt >= 1:
                target_set.add(target_type)

        labels = [0]*len(self.schema2id)
        for target_type in target_set:
            target_type_ids = self.schema2id[target_type]
            labels[target_type_ids] = 1

        return (text, input_ids, token_type_ids, attention_mask, labels)
    
    def __getitem__(self, idx):
        return self.encoder(self.data[idx])
    
    def get_labels(self):
        return self.labels
    
    def get_task_id(self):
        return self._task_id

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_labels = []
        for item in examples:
            (text, input_ids, token_type_ids, attention_mask, labels) = item
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(labels)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_labels = torch.tensor(batch_labels).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_labels
    
class MultiTaskDataset(Dataset):
    """
    https://github.com/namisan/mt-dnn/blob/master/mt_dnn/batcher.py
    """
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, (
                "Duplicate task_id %s" % task_id
            )
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]

    
class MultiTaskBatchSampler(BatchSampler):
    """
    https://github.com/namisan/mt-dnn/blob/master/mt_dnn/batcher.py
    """
    def __init__(
        self,
        datasets,
        batch_size,
        mix_opt,
        extra_task_ratio
    ):
        self._datasets = datasets
        self._batch_size = batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        train_data_list = []
        for dataset in datasets:
            train_data_list.append(
                self._get_shuffled_index_batches(len(dataset), batch_size)
            )
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(
            self._train_data_list, self._mix_opt, self._extra_task_ratio
        )
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(
                min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices))
            )
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices
    
