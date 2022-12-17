# -*- coding: utf-8 -*-

import argparse
import configparser
import logging
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.dataset import ConcatDataset
from transformers import BertConfig, BertTokenizerFast

from nets.unilm_bert import BertForCausalLM
from utils.bert_optimization import BertAdam
from utils.seq2struct_dataloader import (load_duee, load_duie, load_ee_schema,
                                         load_entity, load_entity_schema,
                                         load_ie_schema)
from utils.seq2struct_dataloader_online import (data_generator_flatten_schema,
                                                data_generator_single_schema)
from utils.seq2struct_dataloader_online_general import \
    data_generator_single_schema as data_generator_single_schema_general
from utils.seq2struct_dataloader_online_mrc_qg import (data_generator_mrc,
                                                       load_mrc_schema,
                                                       load_squad_style_data)


class LabelSmoothingLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.reshape(-1)
        model_prob = self.one_hot.float().repeat(target.size(0), 1).to(output.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='none').view(batch_size, num_pos, -1).sum(2)

# 控制随机数种子
def set_seed(seed):
    import random
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

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

args = parse_args() # 从命令行获取
set_seed(42) # 设置随机数种子

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

con = configparser.ConfigParser()

con.read(args.config_file, encoding='utf8')


# 打印配置信息
logger.info("********** Config Info **********")
for sec in con.sections():
    for key, value in dict(con.items(sec)).items():
        logger.info(f"{key}: {value}")
        
args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
# tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext', do_lower_case=True)
tokenizer = BertTokenizerFast.from_pretrained(args_path["tokenizer_path"], do_lower_case=True)


# 根据LOCAL_RANK的值来设置device_id
device_id = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(device_id)
print(f"=> set cuda device = {device_id}")

# device = torch.device("cuda:0")
os.environ["NCCL_BLOCKING_WAIT"] = "1"

print("==> start init_process_group")
dist.init_process_group(
    backend="nccl", init_method="env://", timeout=timedelta(seconds=1000000)
)

rank = int(os.environ.get("RANK", "0"))

if not os.path.exists(args_path['output_path']) and rank == 0:
    # Create a new directory because it does not exist 
    os.makedirs(args_path['output_path'])
    print("The new directory is created!")

# 子进程循环等待主进程构建output_path (避免主进程创建完成前, 子进程进入到logger设置阶段, output_path不存在)
while not os.path.exists(args_path['output_path']):
    print("wait for makedir output_path")
    time.sleep(2)

# log设置
log_file_name = os.path.join(args_path['output_path'], f"train_{rank}.log")
logger.addHandler(logging.FileHandler(log_file_name, 'w'))

device = torch.device("cuda")
num_of_workers = int(os.environ.get("WORLD_SIZE", 1))

print(num_of_workers, device_id, '====num_of_workers----device_id===')

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
            attention_mask_3d = (idxs[:, None, :] <= idxs[:, :, None]).to(dtype=torch.float32)
            model_outputs = self.transformer(input_ids, 
                                             attention_mask=attention_mask_3d, 
                                             token_type_ids=segment_ids)
            return model_outputs # return prediction-scores
        elif mode == "generation":
            model_outputs = self.transformer.generate(
                                            input_ids=input_ids, 
                                            attention_mask=input_mask, 
                                            token_type_ids=segment_ids, 
                                            **kargs) # we need to generate output-scors
        return model_outputs

# 任务参数配置
duie_task_dict = {
    'sep_token':'[SEP]',
    'seg_token':'<S>',
    'group_token':'<T>',
    'start_token':'[CLS]',
    'end_token':'[SEP]',
    'sentinel_token':'[unused{}]',
    'instruction':'信息抽取',
    'sentinel_start_idx':1,
    'add_schema_type':True,
    "greedy_search":True,
    'role_shuffle':True,
    'role_schema_order': True,
    'remove_dup':True,
    'nce_type': True
}

duee_task_dict = {
    'sep_token':'[SEP]',
    'seg_token':'<S>',
    'group_token':'<T>',
    'start_token':'[CLS]',
    'end_token':'[SEP]',
    'sentinel_token':'[unused{}]',
    'instruction':'事件抽取',
    'sentinel_start_idx':1,
    'add_schema_type':True,
    "greedy_search":True,
    'role_shuffle':True,
    'role_schema_order': True,
    'remove_dup':True,
    'nce_type': True
}

entity_task_dict = {
    'sep_token':'[SEP]',
    'seg_token':'<S>',
    'group_token':'<T>',
    'start_token':'[CLS]',
    'end_token':'[SEP]',
    'sentinel_token':'[unused{}]',
    'instruction':'实体抽取',
    'sentinel_start_idx':1,
    'add_schema_type':False,
    "greedy_search":True,
    'role_shuffle':True,
    'remove_dup':True,
    'nce_type': True
}

pretrain_task_dict = {
    'sep_token':'[SEP]',
    'seg_token':'<S>',
    'group_token':'<T>',
    'start_token':'[CLS]',
    'end_token':'[SEP]',
    'sentinel_token':'[unused{}]',
    'instruction':'',
    'sentinel_start_idx':0,
    'add_schema_type':False,
    "greedy_search":True,
    'role_shuffle':True,
    'remove_dup':True,
    'nce_type': True
}


mrc_task_dict = {
    'sep_token':'[SEP]',
    'seg_token':'<S>',
    'group_token':'<T>',
    'start_token':'[CLS]',
    'end_token':'[SEP]',
    'sentinel_token':'[unused{}]',
    'instruction':'答案抽取',
    'sentinel_start_idx':1,
    'add_schema_type':False,
    "greedy_search":True,
    'role_shuffle':True,
    'remove_dup':True
}

# 读取schema文件
schema = []
schema_path_dict = {}
for schema_info in args_path["schema_data"].split(','):
    schema_type, schema_path = schema_info.split(':')
    schema_tuple = tuple(schema_path.split('/')[:-1])
    if schema_tuple not in schema_path_dict:
        schema_path_dict[schema_tuple] = []
    print(schema_type, schema_path, '===schema-path===', schema_tuple)
    
    if 'duie' in schema_type:
        schema.extend(load_ie_schema(schema_path))
        schema_path_dict[schema_tuple] = load_ie_schema(schema_path)
    elif 'duee' in schema_type:
        schema.extend(load_ee_schema(schema_path))
        schema_path_dict[schema_tuple] = load_ee_schema(schema_path)
    elif 'entity' in schema_type:
        schema.extend(load_entity_schema(schema_path))
        schema_path_dict[schema_tuple] = load_entity_schema(schema_path)
    elif 'mrc' in schema_type:
        schema.extend(load_mrc_schema(schema_path))
        schema_path_dict[schema_tuple] = load_mrc_schema(schema_path)
    else:
        print('*****************************************************************')
        print(schema_type, schema_path, '===invalid schema-path===', schema_tuple)
    
total_train_dataset = []
largest_size = 0
dup_factor = 2
for label_index, data_info in enumerate(args_path["train_file"].split(',')):
    data_type, data_path = data_info.split(':')
    data_tuple = tuple(data_path.split('/')[:-1])
    
    if data_tuple in schema_path_dict:
        data_schema = schema_path_dict[data_tuple]
    else:
        print(data_type, data_path, '==invalid data-path==', data_tuple)
        continue

    if 'duie' in data_type:
        load_fn = load_duie
        task_dict = duie_task_dict
        dup_factor = 1
        dataloader_fn = data_generator_single_schema
    elif 'duee' in data_type:
        load_fn = load_duee
        task_dict = duee_task_dict
        dup_factor = 1
        dataloader_fn = data_generator_single_schema
    elif 'duee_general' in data_type:
        load_fn = load_duee
        task_dict = duee_task_dict
        dataloader_fn = data_generator_single_schema_general
    elif 'entity' in data_type:
        load_fn = load_entity
        task_dict = entity_task_dict
        dup_factor = 1
        dataloader_fn = data_generator_single_schema
    elif 'mrc' in data_type:
        load_fn = load_squad_style_data
        task_dict = mrc_task_dict
        dup_factor = 1
        dataloader_fn = data_generator_mrc
    else:
        print('*****************************************************************')
        print(data_type, data_path, '==invalid data-path==', data_tuple)
        continue
            
    print(load_fn, '==load_fn==', data_type, dataloader_fn)
        
    data_list = load_fn(data_path)
    
    # 针对小样本的数据扩增
    if len(data_list) <= 1000:
        data_list *= (20)
    
    train_dataset = dataloader_fn(data_list, tokenizer, max_len=con.getint("para", "maxlen"), 
                                                 schema=data_schema, label=label_index,
                                                task_dict=task_dict, mode='train', add_role_shuffle=False)
    total_train_dataset.append(train_dataset)
    
    if 'entity' in data_type:
        train_entity_dataset = data_generator_flatten_schema(data_list, tokenizer, max_len=con.getint("para", "maxlen"), 
                                                 schema=data_schema, label=label_index,
                                                task_dict=task_dict, mode='train', add_role_shuffle=False)
        total_train_dataset.append(train_entity_dataset)
    
    print(len(train_dataset), '==size of train_dataset==')
    if len(train_dataset) > largest_size:
        largest_size = len(train_dataset)
    
    collate_fn = train_dataset.collate_unilm
    
print(largest_size, '=====largest_size=====', len(total_train_dataset))

train_data = ConcatDataset(total_train_dataset)

if num_of_workers > 1:
    sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, batch_size=con.getint("para", "batch_size"), collate_fn=collate_fn,
                             shuffle=False, drop_last=True)
    data_size = len(sampler)
else:
    train_loader = torch.utils.data.DataLoader(train_data , batch_size=con.getint("para", "batch_size"), collate_fn=collate_fn,
                         shuffle=True)
    data_size = len(train_data)
    


use_amp = True
scaler = None
if use_amp:
    scaler = GradScaler()

def set_optimizer(model, train_steps=None):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=2e-5,
                         warmup=0.01,
                         t_total=train_steps)
    return optimizer

output_path = args_path['output_path']
import os

# 设置模型
net = MyUniLM(config_path=args_path["config_path"], 
              model_path='', 
              eos_token_id=duee_task_dict['end_token'])

# 加载预训练的模型参数
if 'pretrained_output_path' in args_path:
    pretrained_output_path = args_path['pretrained_output_path']

    print(pretrained_output_path, '==pretrained_output_path==')
    ckpt = torch.load(pretrained_output_path, map_location="cpu")
    try:
        net.load_state_dict(ckpt)
    except:
        new_ckpt = {}
        for key in ckpt:
            name = key.split('.')
            new_ckpt[".".join(name[1:])] = ckpt[key]
        net.load_state_dict(new_ckpt)
        print(pretrained_output_path, '==pretrained_output_path==', '======successful loading======', device_id)

net.to(device)
net.train()

net = DistributedDataParallel(net, device_ids=[device_id])
optimizer = set_optimizer(net, train_steps= (int(data_size / con.getint("para", "batch_size")) + 1) * con.getint("para", "epochs"))

# 训练过程
total_loss, total_f1 = 0., 0.
label_smoothing = -1
logger.info('******************* begin training *******************')

for eo in range(con.getint("para", "epochs")):
    
    if num_of_workers > 1:
        sampler.set_epoch(eo)
    
    total_loss = 0
    n_steps = 0
    for idx, batch in enumerate(train_loader):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = batch
        if idx == 0:
            print(tokenizer.decode(batch_token_ids[0]), batch_token_type_ids[0])
        
        batch_token_ids, batch_mask_ids, batch_token_type_ids = batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                model_outputs = net(input_ids=batch_token_ids, input_mask=batch_mask_ids, segment_ids=batch_token_type_ids, mode='train')
                logits = model_outputs[0]

                num_classes = logits.shape[-1]

                pred_logits = logits[:, :-1] # [batch_size, seq-1, num_classes]
                true_labels = batch_token_ids[:, 1:] # [batch_size, seq-1]
                y_mask = batch_token_type_ids[:, 1:].float() # [batch_size, seq-1]

                log_pred_logits = F.log_softmax(pred_logits, dim=-1)
                
                # [batch_size, seq-1]
                if label_smoothing > 0:
                    lm_loss_fn = LabelSmoothingLoss(
                            label_smoothing, num_classes, ignore_index=0, reduction='none')
                else:
                    lm_loss_fn = nn.CrossEntropyLoss(reduction='none')

                # [batch_size, seq-1]
                if label_smoothing > 0:
                    loss_ = lm_loss_fn(log_pred_logits, true_labels)
                else:
                    # sum over label-dim
                    loss_ = lm_loss_fn(pred_logits.transpose(1, 2).float(), true_labels)
    
                loss = torch.sum(loss_ * y_mask) / (1e-12+torch.sum(y_mask))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            model_outputs = net(input_ids=batch_token_ids, input_mask=batch_mask_ids, segment_ids=batch_token_type_ids, mode='train')
            logits = model_outputs[0]

            num_classes = logits.shape[-1]

            pred_logits = logits[:, :-1] # [batch_size, seq-1, num_classes]
            true_labels = batch_token_ids[:, 1:] # [batch_size, seq-1]
            y_mask = batch_token_type_ids[:, 1:].float() # [batch_size, seq-1]

            log_pred_logits = F.log_softmax(pred_logits, dim=-1)

            # [batch_size, seq-1]
            if label_smoothing > 0:
                lm_loss_fn = LabelSmoothingLoss(
                        label_smoothing, num_classes, ignore_index=0, reduction='none')
            else:
                lm_loss_fn = nn.CrossEntropyLoss(reduction='none')

            # [batch_size, seq-1]
            if label_smoothing > 0:
                loss_ = lm_loss_fn(log_pred_logits, true_labels)
            else:
                # sum over label-dim
                loss_ = lm_loss_fn(pred_logits.transpose(1, 2).float(), true_labels)

            loss = torch.sum(loss_ * y_mask) / (1e-12+torch.sum(y_mask))
        
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        n_steps += 1
        if np.mod(idx, 1000) == 0:
            logger.info(" epoch=%d, loss=%.5f, rand=%d ", eo, total_loss/n_steps, int(os.environ.get("RANK", "0")))
        rank = int(os.environ.get("RANK", "0"))
        if np.mod(idx, 10000) == 0 and rank == 0:
            torch.save(net.state_dict(), os.path.join(output_path, 'unilm_mixture.pth.{}'.format(eo)))
            logger.info('==saving path:%s===', output_path)
            
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        torch.save(net.state_dict(), os.path.join(output_path, 'unilm_mixture.pth.{}'.format(eo)))
        logger.info('==saving path:%s===', output_path)
