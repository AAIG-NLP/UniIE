# -*- coding: utf-8 -*-

import argparse
import configparser
import json
import logging
import os
import sys
from collections import namedtuple
from itertools import combinations
from itertools import permutations

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertConfig, BertTokenizerFast

from nets.constrained_decoder import get_end_to_end_prefix_allowed_tokens_fn_hf
from nets.unilm_bert import BertForCausalLM
from utils.seq2struct_dataloader import (load_ee_schema, load_entity_schema,
                                         load_ie_schema)
from utils.seq2struct_decoder import single_schema_decoder


# 控制台参数传入
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help="The config file.", )

    args = parser.parse_args()

    return args


args = parse_args()  # 从命令行获取

# 获取被调用文件的上层路径
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# tokenizer = BertTokenizerFast.from_pretrained(
#     'hfl/chinese-roberta-wwm-ext', do_lower_case=True)

flag = torch.cuda.is_available()
if flag:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# 读取配置文件
con = configparser.ConfigParser()
con.read(args.config_file, encoding='utf8')

tokenizer = BertTokenizerFast.from_pretrained(con.get("paths", "tokenizer_path"), do_lower_case=True)


def dump_instances(instances, output_filename):
    with open(output_filename, 'w', encoding='utf8') as output:
        for instance in instances:
            output.write(json.dumps(instance, ensure_ascii=False) + '\n')


class MyUniLM(nn.Module):
    def __init__(self, config_path, model_path, eos_token_id, **kargs):
        super().__init__()

        self.model_path = model_path
        self.config_path = config_path
        self.eos_token_id = eos_token_id

        self.config = BertConfig.from_pretrained(config_path)
        self.config.is_decoder = True
        self.config.eos_token_id = self.eos_token_id

        self.transformer = BertForCausalLM(config=self.config)

    def forward(self, input_ids, input_mask, segment_ids=None, mode='train', **kargs):
        if mode == "train":
            idxs = torch.cumsum(segment_ids, dim=1)
            attention_mask_3d = (idxs[:, None, :] <= idxs[:, :, None]).to(
                dtype=torch.float32)
            model_outputs = self.transformer(input_ids,
                                             attention_mask=attention_mask_3d,
                                             token_type_ids=segment_ids)
            return model_outputs  # return prediction-scores
        elif mode == "generation":
            model_outputs = self.transformer.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                **kargs)  # we need to generate output-scors
        return model_outputs


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


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


def extract(item, task_dict, all_schema_dict, target_type, max_len):
    text = item['text']

    # generate instruction input-ids
    instruction_text = task_dict['start_token'] + \
        task_dict['instruction'] + task_dict['sep_token']
    encoder_instruction_text = tokenizer(
        instruction_text, return_offsets_mapping=True, add_special_tokens=False)
    instruction_input_ids = encoder_instruction_text["input_ids"]
    # RoBERTa不需要NSP任务
    instruction_token_type_ids = encoder_instruction_text["token_type_ids"]
    # RoBERTa不需要NSP任务
    instruction_attention_mask = encoder_instruction_text["attention_mask"]

    # generate input-ids
    encoder_text = tokenizer(
        text, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
    input_ids = encoder_text["input_ids"]
    token_type_ids = encoder_text["token_type_ids"]  # RoBERTa不需要NSP任务
    attention_mask = encoder_text["attention_mask"]

    # generate schema
    offset_mapping = encoder_text.offset_mapping
    schema_dict = all_schema_dict[target_type]
    if task_dict['add_schema_type']:
        schema_strings = target_type + task_dict['sep_token']
    else:
        schema_strings = ''
    for role in schema_dict['role2sentinel']:
        schema_strings += role + \
            schema_dict['role2sentinel'][role]  # role sentinel

    schema_strings += task_dict['sep_token']

    encoder_schema_text = tokenizer(schema_strings, return_offsets_mapping=True, max_length=max_len,
                                    truncation=False, add_special_tokens=False)
    schema_input_ids = encoder_schema_text["input_ids"]
    # RoBERTa不需要NSP任务
    schema_token_type_ids = encoder_schema_text["token_type_ids"]
    schema_attention_mask = encoder_schema_text["attention_mask"]

    output_list = []
    doc_spans = slide_window(input_ids, max_len, 16, offset=0)
    for doc_span in doc_spans:
        span_start = doc_span.start
        span_end = doc_span.start + doc_span.length

        span_input_ids = input_ids[span_start:span_end] + tokenizer(
            task_dict['sep_token'], add_special_tokens=False)['input_ids']

        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        output_list.append((offset_mapping, instruction_input_ids,
                           span_input_ids, schema_input_ids, input_ids))
    return output_list


def predict_seq2struct(net, decoder, data, task_dict, schema_dict, con):
    decoded_list = []
    # 遍历抽取
    for target_type in data['schema_type']:
        output_list = extract(data, task_dict, schema_dict,
                              target_type, max_len=256)
        for output_input in output_list:
            (offset_mapping, instruction_input_ids, input_ids,
             schema_input_ids, ori_input_ids) = output_input
            text = data['text']

            query_token_ids = instruction_input_ids + input_ids + schema_input_ids


            prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(
                input_ids, task_dict, tokenizer)

            batch_token_ids = torch.tensor(
                query_token_ids).long().unsqueeze(0).to(device)
            batch_mask_ids = torch.tensor(
                [1] * len(query_token_ids)).long().unsqueeze(0).to(device)
            batch_token_type_ids = torch.tensor(
                [0] * len(query_token_ids)).long().unsqueeze(0).to(device)

            model_outputs = net(input_ids=batch_token_ids, input_mask=batch_mask_ids, segment_ids=batch_token_type_ids,
                                mode='generation',
                                output_scores=True, do_sample=False, max_length=1024, num_beams=2, return_dict_in_generate=True,
                                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
                                )

            decode_output_list = decoder.single_schema_decode(
                text, offset_mapping, target_type, model_outputs, query_token_ids, ori_input_ids, mode='unilm')

            for output in decode_output_list:
                decoded_list.append(output)
    return decoded_list


def build_schema(task_dict, schema_dict_list, con):
    sentinel_token = task_dict['sentinel_token']
    sentinel_start_idx = task_dict['sentinel_start_idx']

    all_schema_dict = {}
    for schema_dict in schema_dict_list:
        if schema_dict['type'] not in all_schema_dict:
            all_schema_dict[schema_dict['type']] = {
                'role2sentinel': {},
                'sentinel2role': {},
                'role2type': {},
                'type2role': {},
                'role_index': 0
            }
        role_index = all_schema_dict[schema_dict['type']]['role_index']
        for _, role_dict in enumerate(schema_dict['role_list']):
            role_type = role_dict['type'] + role_dict['role']
            if role_type in all_schema_dict[schema_dict['type']]['role2sentinel']:
                continue
            all_schema_dict[schema_dict['type']]['role2sentinel'][role_type] = sentinel_token.format(
                role_index+sentinel_start_idx)
            all_schema_dict[schema_dict['type']]['sentinel2role'][sentinel_token.format(
                role_index+sentinel_start_idx)] = role_type
            all_schema_dict[schema_dict['type']
                            ]['role2type'][role_dict['role']] = role_type
            all_schema_dict[schema_dict['type']
                            ]['type2role'][role_type] = role_dict['role']
            role_index += 1
        all_schema_dict[schema_dict['type']]['role_index'] = role_index

    decoder = single_schema_decoder(tokenizer, 256, schema_dict_list, label=0,
                                    task_dict=task_dict, mode='train')
    return all_schema_dict, decoder


def load_ckpt(net, path, device):
    # ckpt = torch.load(self.args_path[''],  map_location="cpu")
    ckpt = torch.load(path,  map_location=device)
    try:
        net.load_state_dict(ckpt)
    except:
        new_ckpt = {}
        for key in ckpt:
            name = key.split('.')
            new_ckpt[".".join(name[1:])] = ckpt[key]
        net.load_state_dict(new_ckpt)
    net.eval()

    return net


class InferModel(object):
    def __init__(self, con):
        self.con = con

        self.args_path = dict(dict(self.con.items('paths')),
                              **dict(self.con.items("para")))
        
        # 打印配置信息
        logger.info("********** Config Info **********")
        for sec in self.con.sections():
            for key, value in dict(self.con.items(sec)).items():
                logger.info(f"{key}: {value}")

        # for key in self.args_path.keys():
        #     if key not in ['schema_data', 'output_epoch']:
        #         self.args_path[key] = os.path.join(cur_dir_path, self.args_path[key])

        self.duie_task_dict = {
            'sep_token': '[SEP]',
            'seg_token': '<S>',
            'group_token': '<T>',
            'start_token': '[CLS]',
            'end_token': '[SEP]',
            'sentinel_token': '[unused{}]',
            'instruction': '信息抽取',
            'sentinel_start_idx': 1,
            'add_schema_type': True
        }

        self.duee_task_dict = {
            'sep_token': '[SEP]',
            'seg_token': '<S>',
            'group_token': '<T>',
            'start_token': '[CLS]',
            'end_token': '[SEP]',
            'sentinel_token': '[unused{}]',
            'instruction': '事件抽取',
            'sentinel_start_idx': 1,
            'add_schema_type': True
        }

        self.entity_task_dict = {
            'sep_token': '[SEP]',
            'seg_token': '<S>',
            'group_token': '<T>',
            'start_token': '[CLS]',
            'end_token': '[SEP]',
            'sentinel_token': '[unused{}]',
            'instruction': '实体抽取',
            'sentinel_start_idx': 1,
            'add_schema_type': False
        }

        self.schema = []
        self.schema_path_dict = {}
        for schema_info in self.args_path["schema_data"].split(','):
            schema_type, schema_path = schema_info.split(':')
            schema_path = os.path.join(cur_dir_path, schema_path)
            schema_tuple = tuple(schema_path.split('/')[:-1])
            if schema_type not in self.schema_path_dict:
                self.schema_path_dict[schema_type] = []
            # print(schema_type, schema_path, '===schema-path===', schema_type)
            if 'duie' in schema_type:
                self.schema.extend(load_ie_schema(schema_path))
                self.schema_path_dict[schema_type] = load_ie_schema(
                    schema_path)
            elif 'duee' in schema_type:
                self.schema.extend(load_ee_schema(schema_path))
                self.schema_path_dict[schema_type] = load_ee_schema(
                    schema_path)
            elif 'entity' in schema_type:
                self.schema.extend(load_entity_schema(schema_path))
                self.schema_path_dict[schema_type] = load_entity_schema(
                    schema_path)

        # print(self.args_path)
        # 加载各个单模型
        # print(self.args_path["config_path"])
        self.net = MyUniLM(
            config_path=self.args_path["config_path"], model_path='', eos_token_id=tokenizer.sep_token_id)
        self.net.to(device)
        self.net = load_ckpt(self.net, os.path.join(
            self.args_path['output_path'], "unilm_mixture.pth.19"), device)

    # 遍历类型进行推理示例
    def predict(self):
        
        test_schema_type = self.args_path['test_schema_type']

        if "duee" in test_schema_type: # 事件抽取推理
            task_dict = self.duee_task_dict
        elif "duie" in test_schema_type: # 关系元组推理
            task_dict = self.duie_task_dict
        elif "entity" in test_schema_type: # 实体推理
            task_dict = self.entity_task_dict

        # 读取训练数据
        with open(self.args_path['test_file'], 'r') as frobj:
            pred_list = []

            for idx, line in tqdm(enumerate(frobj)):
                content = json.loads(line.strip())
                tmp_dict = {
                    'text': content['text'],
                    'entity': [],
                    'relation': [],
                    'event': []
                }

                all_schema_dict, decoder = build_schema(
                    task_dict, self.schema_path_dict[test_schema_type], self.con)

                content['schema_type'] = list(all_schema_dict.keys())
                decoded_list = predict_seq2struct(
                    self.net, decoder, content, task_dict, all_schema_dict, self.con)
                # print(decoded_list)
                
                # 根据decode_list解析出对应的结果
                if "duee" in test_schema_type:
                    # 将事件元组拆分成三元组进行结果存储, 用于业务评估
                    triple_set = set()
                    for event in decoded_list:
                        for t in event:
                            if t not in triple_set: triple_set.add(t)
                    
                    tmp_dict['event'] = list(triple_set)
                elif "duie" in test_schema_type:
                    # 关系元组
                    for rel in decoded_list:
                        tmp_dict['relation'].append({
                            "type": rel[0][0],
                            "subject_type": rel[0][1],
                            "subject_text": rel[0][2],
                            "object_type": rel[1][1],
                            "object_text": rel[1][2]
                        })
                elif "entity" in test_schema_type:
                    # 实体元组
                    for entity in decoded_list:
                        entity = entity[0] # 由于分组的概念，外面会多一层列表
                        tmp_dict['entity'].append({
                            'type': entity[0],
                            'text': entity[2]
                        })
                
                pred_list.append(tmp_dict)

            dump_instances(pred_list, self.args_path['prediction_output_path'])
        


    # (CCKS2022 通用信息抽取 提交结果样式)
    def predict_ccks(self):

        schema_mapping_dict = {
            '金融信息': 'duee_fin',
            '影视情感': 'duie_asa',
            '人生信息': 'duie_life',
            '机构信息': 'duie_org',
            '体育竞赛': 'duee',
            '灾害意外': 'duee',
            '金融监管': 'duie_fin_monitor',
            '流调信息': 'duee_dieaese',
            '金融舆情': 'duee_fin_news',
            '医患对话': 'duie_asa_medical'
        }

        valid_schema_type = {
            'duee':['爆炸', '车祸', '地震', '洪灾', '起火', '坠机', '坍/垮塌', 
                '袭击', '坠机', '夺冠', '晋级', '禁赛', '胜负', '退赛', '退役']
            }

        entity_type_mapping = {
            '人物':'人物',
            '组织机构':'组织机构',
            '地理位置':'地理位置',
            '疾病':'疾病',
            '医学检查':'检验检查',
            '临床表现':'症状',
            '症状':'症状',
            '疾病': '疾病',
            '检验检查': '检验检查',
            '手术': '手术',
            '药物': '药物'
        }

        medical_pair = {
            ('症状', '部位'):'',
            ('疾病', '部位'):'',
            ('手术', '部位'):'',
            ('检验检查', '部位'):''
        }
        
        et_dict = {}
        # print(self.args_path)
        prediction_output_path = os.path.join(self.args_path['output_path'], "submit_result.json")
        
        # 该文件已包含预测出的子事件类型, 同时待预测文件的文本也包含在其中, 因此不需要加载test_path
        schema_type_path = self.args_path['prediction_schema_path'] 

        with open(prediction_output_path, 'w') as fwobj:
            with open(schema_type_path, 'r') as frobj: # 给定类型的文件, 字段为 schema_type
                for line in tqdm(frobj):
                    content = json.loads(line.strip())
                    schema_type = content['schema']
                    if 'duie' in schema_mapping_dict[schema_type]:
                        task_dict = self.duie_task_dict
                    elif 'duee' in schema_mapping_dict[schema_type]:
                        task_dict = self.duee_task_dict
                    all_schema_dict, decoder = build_schema(
                        task_dict, self.schema_path_dict[schema_mapping_dict[schema_type]], self.con)
                    tmp_dict = {
                        'id': content['id'],
                        'entity': [],
                        'relation': [],
                        'event': []
                    }

                    # 设置遍历推理的类型, 覆盖给定的类型字段
                    if schema_mapping_dict[schema_type] == 'duee':
                        content['schema_type'] = valid_schema_type['duee']
                    else:
                        content['schema_type'] = list(all_schema_dict.keys())

                    if 'duie' in schema_mapping_dict[schema_type] or 'asa' in schema_mapping_dict[schema_type]:
                        # content['schema_type'] = []

                        if content['id'] in et_dict:
                            for schema_type in et_dict[content['id']]:
                                content['schema_type'].append(schema_type)
                            content['schema_type'] = list(set(content['schema_type']))

                        decoded_list = predict_seq2struct(
                            self.net, decoder, content, task_dict, all_schema_dict, self.con)
                        for decoded in decoded_list:
                            """
                            [('正向情感', '意见对象', '阿婴'), ('正向情感', '情感表达', '挺刺激')]
                            """

                            if schema_mapping_dict[schema_type] in ['duie_asa_medical']: # duie医药推理
                                if len(decoded) >= 3:
                                    """
                                    ('部位', '症状', '尿失禁'), ('部位', '部位', '前列腺'), ('部位', '手术', '前列腺增生 电切术')
                                    """
                                    if decoded[0][0] in ['部位']:
                                        decoded_combination = list(
                                            combinations(decoded, 2))
                                        for combine_decoded in decoded_combination:
                                            if (combine_decoded[0][1], combine_decoded[1][1]) in medical_pair:
                                                sub_dict = {
                                                    'type': combine_decoded[0][0],
                                                    'args': [
                                                    ]

                                                }
                                                for output in [combine_decoded[0], combine_decoded[1]]:
                                                    sub_dict['args'].append({
                                                        'type': output[1],
                                                        'text': output[2],
                                                    })
                                                    if schema_mapping_dict[schema_type] in ['duie_asa_medical', 'duie_life', 'duie_org']:
                                                        if output[1] in entity_type_mapping:
                                                            tmp_dict['entity'].append(
                                                                {'type': entity_type_mapping[output[1]], 'text': output[2]})
                                                tmp_dict['relation'].append(sub_dict)
                                            elif (combine_decoded[1][1], combine_decoded[0][1]) in medical_pair:
                                                sub_dict = {
                                                    'type': combine_decoded[1][0],
                                                    'args': [
                                                    ]
                                                }
                                                for output in [combine_decoded[1], combine_decoded[0]]:
                                                    sub_dict['args'].append({
                                                        'type': output[1],
                                                        'text': output[2],
                                                    })
                                                    if schema_mapping_dict[schema_type] in ['duie_asa_medical', 'duie_life', 'duie_org']:
                                                        if output[1] in entity_type_mapping:
                                                            tmp_dict['entity'].append(
                                                                {'type': entity_type_mapping[output[1]], 'text': output[2]})
                                                tmp_dict['relation'].append(sub_dict)
                                else:
                                    if len(decoded) == 2:
                                        decoded_combination = list(permutations(decoded, 2)) # 做一个全排列, 解决p相同而s、o不同的问题
                                        for combine_decoded in decoded_combination:
                                            if (combine_decoded[0][1], combine_decoded[1][1]) in medical_pair:
                                                sub_dict = {
                                                    'type':combine_decoded[0][0],
                                                    'args':[
                                                    ]
                                                }
                                                for output in [combine_decoded[0], combine_decoded[1]]:
                                                    sub_dict['args'].append({
                                                        'type':output[1],
                                                        'text':output[2],
                                                    })

                                                if schema_mapping_dict[schema_type] in ['duie_asa_medical', 'duie_life', 'duie_org']:
                                                    if output[1] in entity_type_mapping:
                                                        tmp_dict['entity'].append(
                                                            {'type': entity_type_mapping[output[1]], 'text': output[2]})
                                                tmp_dict['relation'].append(sub_dict)
                            else:
                                if len(decoded) == 2:
                                    sub_dict = {
                                        'type': decoded[0][0],
                                        'args': [
                                        ]
                                    }
                                    for output in decoded:
                                        sub_dict['args'].append({
                                            'type': output[1],
                                            'text': output[2],
                                        })
                                        if schema_mapping_dict[schema_type] in ['duie_asa_medical', 'duie_life', 'duie_org']:
                                            if output[1] in entity_type_mapping:
                                                tmp_dict['entity'].append(
                                                    {'type': entity_type_mapping[output[1]], 'text': output[2]})

                                    tmp_dict['relation'].append(sub_dict)
                        fwobj.write(json.dumps(tmp_dict, ensure_ascii=False)+'\n')
                    elif 'duee' in schema_mapping_dict[schema_type]:
                        if content['id'] in et_dict:
                            for schema_type in et_dict[content['id']]:
                                content['schema_type'].append(schema_type)
                            content['schema_type'] = list(set(content['schema_type']))

                        decoded_list = predict_seq2struct(
                            self.net, decoder, content, task_dict, all_schema_dict, self.con)
                        for decoded in decoded_list:
                            """
                            [('夺冠', '触发词', '金牌'), ('夺冠', '时间', '今天上午'), ('夺冠', '冠军', '中谷爱凌'), ('夺冠', '夺冠赛事', '北京冬奥会自由式滑雪女子大跳台决赛')]
                            """
                            if decoded:
                                event_dict = {
                                    "type": decoded[0][0],
                                    'text': "",
                                    "args": []

                                }
                                for item in decoded:
                                    if item[1] == u'触发词':
                                        event_dict['text'] = item[2]
                                    else:
                                        event_dict['args'].append({
                                            'type': item[1],
                                            'text': item[2]
                                        })
                                tmp_dict['event'].append(event_dict)
                        fwobj.write(json.dumps(tmp_dict, ensure_ascii=False)+'\n')

                print("==output-final-path==", prediction_output_path)





# 调用模块进行预测
infer_model = InferModel(con=con)

# infer_model.predict()
infer_model.predict_ccks()
