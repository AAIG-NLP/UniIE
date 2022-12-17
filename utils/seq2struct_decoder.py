
from utils.flashtext import KeywordProcessor
from collections import namedtuple
from collections import OrderedDict
import torch

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class single_schema_decoder(object):
    def __init__(self, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train'):
        
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
                
        
        self.trie_tree = KeywordProcessor()
        for token in [self.sep_token, self.seg_token, self.group_token, self.start_token, self.end_token]:
            tokens = self.tokenizer.tokenize(token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(tokens), token)
            
        for i in range(1, 100):
            sentinel_token = self.sentinel_token.format(i)
            sentinel_tokens = self.tokenizer.tokenize(sentinel_token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(sentinel_tokens), sentinel_token)
            
    def single_schema_decode(self, text, token2char_span_mapping, schema_type, result_dict, query_token_ids, ori_input_ids, mode='unilm', search_mode='token_id'):
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        
        if isinstance(result_dict.sequences, torch.Tensor):
            token_ids = result_dict.sequences[0].cpu().numpy().tolist()
        else:
            token_ids = result_dict.sequences[0]
            
        # print(self.tokenizer.decode(token_ids))
        
        if mode == 'unilm':
            input_len = len(query_token_ids)
            generated_token_ids = token_ids[input_len:]
        elif mode == 'seq2seq_no_decoder_input':
            generated_token_ids = token_ids[1:]
            
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        schema_dict = self.schema_dict[schema_type]
            
        group_output_list = []
        start_index = 0
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break

            pred_tokens = []
            group_tokens = []

            if word in [self.group_token]:
                group_token_ids = generated_token_ids[start_index:start] # value-tokens + tole_type-tokens
                start_index = end
                group_output_list.append(group_token_ids)
        
        schema_output_list = []
        for group_token_ids in group_output_list:
            keywords = self.trie_tree.extract_keywords(group_token_ids, span_info=True)
            start_index = 0
            pred_tuple = []
            for keyword in keywords:
                word = keyword[0]
                start = keyword[1]
                end = keyword[2]
                if word in schema_dict['sentinel2role']:
                    role_type = schema_dict['sentinel2role'][word]
                    role = schema_dict['type2role'][role_type]
                    pred_token_ids = group_token_ids[start_index:start] # value-tokens + tole_type-tokens
                    start_index = end
                    
                    if search_mode == 'token_id':
                        idx = search(pred_token_ids, ori_input_ids)
                        if idx != -1:
                            try:
                                start, end = new_span[idx][0], new_span[idx+len(pred_token_ids)-1][-1] + 1
                                pred_tuple.append((schema_type, role, text[start:end]))
                            except:
                                continue
                    elif search_mode == 'string':
                        pred_string = self.tokenizer.decode(pred_token_ids)
                        idx = search(pred_string, text)
                        if idx != -1:
                            pred_tuple.append((schema_type, role, pred_string))
                    
            schema_output_list.append(pred_tuple)
        return schema_output_list
    
class single_schema_decoder_pos(single_schema_decoder):
    def __init__(self, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train'):
        
        super().__init__(tokenizer, max_len, schema, label, 
                 task_dict, mode)
        
    def single_schema_decode(self, text, token2char_span_mapping, schema_type, result_dict, query_token_ids, ori_input_ids, mode='unilm', search_mode='token_id'):
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        
        if isinstance(result_dict.sequences, torch.Tensor):
            token_ids = result_dict.sequences[0].cpu().numpy().tolist()
        else:
            token_ids = result_dict.sequences[0]
            
        # print(self.tokenizer.decode(token_ids))
        
        if mode == 'unilm':
            input_len = len(query_token_ids)
            generated_token_ids = token_ids[input_len:]
        elif mode == 'seq2seq_no_decoder_input':
            generated_token_ids = token_ids[1:]
            
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        schema_dict = self.schema_dict[schema_type]
            
        group_output_list = []
        start_index = 0
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break

            pred_tokens = []
            group_tokens = []

            if word in [self.group_token]:
                group_token_ids = generated_token_ids[start_index:start] # value-tokens + tole_type-tokens
                start_index = end
                group_output_list.append(group_token_ids)
        
        schema_output_list = []
        for group_token_ids in group_output_list:
            keywords = self.trie_tree.extract_keywords(group_token_ids, span_info=True)
            start_index = 0
            pred_tuple = []
            for keyword in keywords:
                word = keyword[0]
                start = keyword[1]
                end = keyword[2]
                if word in schema_dict['sentinel2role']:
                    role_type = schema_dict['sentinel2role'][word]
                    role = schema_dict['type2role'][role_type]
                    pred_token_ids = group_token_ids[start_index:start] # value-tokens + tole_type-tokens
                    start_index = end
                    
                    if search_mode == 'token_id':
                        idx = search(pred_token_ids, ori_input_ids)
                        if idx != -1:
                            try:
                                start, end = new_span[idx][0], new_span[idx+len(pred_token_ids)-1][-1] + 1
                                pred_tuple.append((schema_type, role, text[start:end], (start, end)))
                            except:
                                continue
                    elif search_mode == 'string':
                        pred_string = self.tokenizer.decode(pred_token_ids)
                        idx = search(pred_string, text)
                        if idx != -1:
                            pred_tuple.append((schema_type, role, pred_string))
                    
            schema_output_list.append(pred_tuple)
        return schema_output_list 

    
class element_decoder(single_schema_decoder):
    def __init__(self, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train'):
        
        super().__init__(tokenizer, max_len, schema, label, 
                 task_dict, mode)
        
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
        
        self.trie_tree = KeywordProcessor()
        for token in [self.sep_token, self.seg_token, self.group_token, self.start_token, self.end_token]:
            tokens = self.tokenizer.tokenize(token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(tokens), token)
            
        for i in range(1, 100):
            sentinel_token = self.sentinel_token.format(i)
            sentinel_tokens = self.tokenizer.tokenize(sentinel_token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(sentinel_tokens), sentinel_token)
            
    def single_schema_decode(self, text, token2char_span_mapping, schema_type, result_dict, query_token_ids, ori_input_ids, mode='unilm'):
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        
        if isinstance(result_dict.sequences, torch.Tensor):
            token_ids = result_dict.sequences[0].cpu().numpy().tolist()
        else:
            token_ids = result_dict.sequences[0]
            
        # print(self.tokenizer.decode(token_ids))
        
        if mode == 'unilm':
            input_len = len(query_token_ids)
            generated_token_ids = token_ids[input_len:]
        elif mode == 'seq2seq':
            generated_token_ids = token_ids
            
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        schema_dict = self.schema_dict
    
        start_index = 0
        schema_output_list = []
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break

            if word in schema_dict['sentinel2role']:
                role_type = schema_dict['sentinel2role'][word]
                role = schema_dict['type2role'][role_type]
                pred_token_ids = generated_token_ids[start_index:start] # value-tokens + tole_type-tokens
                start_index = end
                idx = search(pred_token_ids, ori_input_ids)
                if idx != -1:
                    try:
                        start, end = new_span[idx][0], new_span[idx+len(pred_token_ids)-1][-1] + 1
                        schema_output_list.append((schema_type, role, text[start:end]))
                    except:
                        continue
                        
        # print(schema_output_list)
                
        return schema_output_list
    
class schema_type_decoder(single_schema_decoder):
    def __init__(self, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train'):
        
        super().__init__(tokenizer, max_len, schema, label, 
                 task_dict, mode)
        
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
        self.schema_dict['role2sentinel']['类型'] = self.sentinel_token.format(self.sentinel_start_idx)
        self.schema_dict['sentinel2role'][self.sentinel_token.format(self.sentinel_start_idx)] = '类型'
        self.schema_dict['role2type']['类型'] = '类型'
        self.schema_dict['type2role']['类型'] = '类型'
        
        self.trie_tree = KeywordProcessor()
        for token in [self.sep_token, self.seg_token, self.group_token, self.start_token, self.end_token]:
            tokens = self.tokenizer.tokenize(token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(tokens), token)
            
        for i in range(1, 100):
            sentinel_token = self.sentinel_token.format(i)
            sentinel_tokens = self.tokenizer.tokenize(sentinel_token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(sentinel_tokens), sentinel_token)
            
    def single_schema_decode(self, text, token2char_span_mapping, schema_type, result_dict, query_token_ids, ori_input_ids, mode='unilm'):
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        
        if isinstance(result_dict.sequences, torch.Tensor):
            token_ids = result_dict.sequences[0].cpu().numpy().tolist()
        else:
            token_ids = result_dict.sequences[0]
            
        # print(self.tokenizer.decode(token_ids))
        
        if mode == 'unilm':
            input_len = len(query_token_ids)
            generated_token_ids = token_ids[input_len:]
        elif mode == 'seq2seq':
            generated_token_ids = token_ids
            
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        schema_dict = self.schema_dict
    
        start_index = 0
        schema_output_list = []
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break

            if word in schema_dict['sentinel2role']:
                role_type = schema_dict['sentinel2role'][word]
                role = schema_dict['type2role'][role_type]
                pred_token_ids = generated_token_ids[start_index:start] # value-tokens + tole_type-tokens
                start_index = end
                idx = search(pred_token_ids, ori_input_ids)
                if idx != -1:
                    try:
                        start, end = new_span[idx][0], new_span[idx+len(pred_token_ids)-1][-1] + 1
                        schema_output_list.append((schema_type, role, text[start:end]))
                    except:
                        continue
                        
        # print(schema_output_list)
                
        return schema_output_list
            

class element_group_decoder(single_schema_decoder):
    def __init__(self, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train'):
        
        super().__init__(tokenizer, max_len, schema, label, 
                 task_dict, mode)
        
    def single_schema_decode(self, text, token2char_span_mapping, schema_type, result_dict, query_token_ids, ori_input_ids, element_schema_dict, mode='unilm'):
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        
        if isinstance(result_dict.sequences, torch.Tensor):
            token_ids = result_dict.sequences[0].cpu().numpy().tolist()
        else:
            token_ids = result_dict.sequences[0]
            
        # print(self.tokenizer.decode(token_ids))
        
        if mode == 'unilm':
            input_len = len(query_token_ids)
            generated_token_ids = token_ids[input_len:]
        elif mode == 'seq2seq':
            generated_token_ids = token_ids
            
        
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        schema_dict = self.schema_dict[schema_type]
    
        group_output_list = []
        start_index = 0
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break

            pred_tokens = []
            group_tokens = []

            if word in [self.group_token]:
                group_token_ids = generated_token_ids[start_index:start] # value-tokens + tole_type-tokens
                start_index = end
                group_output_list.append(group_token_ids)
                
        # [unused51][unused1]<S>[unused51][unused1]<S><T>
        schema_output_list = []
        for group_token_ids in group_output_list:
            keywords = self.trie_tree.extract_keywords(group_token_ids, span_info=True)
            start_index = 0
            pred_tuple = []
            for keyword in keywords:
                word = keyword[0]
                start = keyword[1]
                end = keyword[2]
                
                if word in [self.seg_token]:
                    element_role_words = group_token_ids[start_index:start]
                    start_index = end
                    element_keywords = self.trie_tree.extract_keywords(element_role_words, span_info=True)
                    
                    element = element_keywords[0]
                    if element in element_schema_dict['sentinel2element']:
                        real_element = element_schema_dict['sentinel2element'][element]
                    else:
                        real_element = ''
                    role = element_keywords[1]
                    if role in schema_dict['sentinel2role']:
                        real_role = schema_dict['type2role'][role_type]
                    else:
                        real_role = ''
                    
                    if real_role and real_element:
                        pred_tuple.append((schema_type, real_role, real_element))
            schema_output_list.append(pred_tuple)
        return schema_output_list
        
class single_schema_element_and_group_decoder(object):
    def __init__(self, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train'):
        
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
        self.element_start_idx = self.task_dict['element_start_idx']
        
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
                
        
        self.element_dict = {}
        for i in range(1, 100):
            self.element_dict[self.sentinel_token.format(i)] = self.sentinel_token.format(i)
                
        self.trie_tree = KeywordProcessor()
        for token in [self.sep_token, self.seg_token, self.group_token, self.start_token, self.end_token]:
            tokens = self.tokenizer.tokenize(token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(tokens), token)
            
        for i in range(1, 100):
            sentinel_token = self.sentinel_token.format(i)
            sentinel_tokens = self.tokenizer.tokenize(sentinel_token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(sentinel_tokens), sentinel_token)
            
    def single_schema_decode(self, text, token2char_span_mapping, schema_type, result_dict, query_token_ids, ori_input_ids, mode='unilm'):
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        
        if isinstance(result_dict.sequences, torch.Tensor):
            token_ids = result_dict.sequences[0].cpu().numpy().tolist()
        else:
            token_ids = result_dict.sequences[0]
            
        # print(self.tokenizer.decode(token_ids))
        
        if mode == 'unilm':
            input_len = len(query_token_ids)
            generated_token_ids = token_ids[input_len:]
        elif mode == 'seq2seq':
            generated_token_ids = token_ids
            
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        schema_dict = self.schema_dict[schema_type]
            
        element_output_list = []
        group_output_list = []
        start_index = 0
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break

            if word in [self.sentinel_token.format(99)]:
                element_token_ids = generated_token_ids[start_index:start] # value-tokens + tole_type-tokens
                start_index = end
                element_output_list.append(element_token_ids)
            elif word in [self.group_token]:
                element_group_token_ids = generated_token_ids[start_index:start] # value-tokens + tole_type-tokens
                group_output_list.append(element_group_token_ids)
                start_index = end
                
        element_schema_dict = {
                'element2sentinel':OrderedDict({}),
                'sentinel2element':OrderedDict({}),
                'element2pos':OrderedDict({})
        }
        
        elements = set()
        for element_token_ids in element_output_list:
            keywords = self.trie_tree.extract_keywords(element_token_ids, span_info=True)
            start_index = 0
            for keyword in keywords:
                word = keyword[0]
                start = keyword[1]
                end = keyword[2]
                if word in self.element_dict:
                    pred_token_ids = element_token_ids[start_index:start] # value-tokens + tole_type-tokens
                    start_index = end
                    idx = search(pred_token_ids, ori_input_ids)
                    if idx != -1:
                        try:
                            start, end = new_span[idx][0], new_span[idx+len(pred_token_ids)-1][-1] + 1
                            elements.add((text[start:end], word)) # words, sentinel-word
                        except:
                            continue

        for element in elements:
            element_schema_dict['element2sentinel'][element[0]] = element[1]
            element_schema_dict['sentinel2element'][element[1]] = element[0]
        
        schema_output_list = []
        for group_token_ids in group_output_list:
            keywords = self.trie_tree.extract_keywords(group_token_ids, span_info=True)
            start_index = 0
            
            pred_tuple = []
              
            for keyword in keywords:
                word = keyword[0]
                start = keyword[1]
                end = keyword[2]
                if word in element_schema_dict['sentinel2element']:
                    pred_tuple.append(('element', element_schema_dict['sentinel2element'][word]))
                elif word in schema_dict['sentinel2role']:
                    role_type = schema_dict['sentinel2role'][word]
                    role = schema_dict['type2role'][role_type]
                    pred_tuple.append(('role', role))
                    
            pred_pair = []
            for idx in range(0, len(pred_tuple), 2):
                pred_tuple.append((schema_type, role, text[start:end]))
                pred_pair.append((schema_type, pred_tuple[idx+1][1], pred_tuple[idx][1]))
            
            schema_output_list.append(pred_pair)
                    
        return schema_output_list
    
class single_schema_decoder_cardinality(object):
    def __init__(self, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train'):
        
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
                
        
        self.trie_tree = KeywordProcessor()
        for token in [self.sep_token, self.seg_token, self.group_token, self.start_token, self.end_token]:
            tokens = self.tokenizer.tokenize(token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(tokens), token)
            
        for i in range(1, 100):
            sentinel_token = self.sentinel_token.format(i)
            sentinel_tokens = self.tokenizer.tokenize(sentinel_token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(sentinel_tokens), sentinel_token)
            
    def single_schema_decode(self, text, token2char_span_mapping, schema_type, result_dict, query_token_ids, ori_input_ids, mode='unilm', search_mode='token_id'):
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        
        if isinstance(result_dict.sequences, torch.Tensor):
            token_ids = result_dict.sequences[0].cpu().numpy().tolist()
        else:
            token_ids = result_dict.sequences[0]
            
        # print(self.tokenizer.decode(token_ids))
        
        if mode == 'unilm':
            input_len = len(query_token_ids)
            generated_token_ids = token_ids[input_len:]
        elif mode == 'seq2seq_no_decoder_input':
            generated_token_ids = token_ids[1:]
            
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        schema_dict = self.schema_dict[schema_type]
            
        group_output_list = []
        start_index = 0
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break
                
            if word in [self.seg_token]:
                generated_token_ids = generated_token_ids[end:]
                
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        group_output_list = []
        start_index = 0
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break
    
            pred_tokens = []
            group_tokens = []

            if word in [self.group_token]:
                group_token_ids = generated_token_ids[start_index:start] # value-tokens + tole_type-tokens
                start_index = end
                group_output_list.append(group_token_ids)
        
        schema_output_list = []
        for group_token_ids in group_output_list:
            keywords = self.trie_tree.extract_keywords(group_token_ids, span_info=True)
            start_index = 0
            pred_tuple = []
            for keyword in keywords:
                word = keyword[0]
                start = keyword[1]
                end = keyword[2]
                if word in schema_dict['sentinel2role']:
                    role_type = schema_dict['sentinel2role'][word]
                    role = schema_dict['type2role'][role_type]
                    pred_token_ids = group_token_ids[start_index:start] # value-tokens + tole_type-tokens
                    start_index = end
                    
                    if search_mode == 'token_id':
                        idx = search(pred_token_ids, ori_input_ids)
                        if idx != -1:
                            try:
                                start, end = new_span[idx][0], new_span[idx+len(pred_token_ids)-1][-1] + 1
                                pred_tuple.append((schema_type, role, text[start:end]))
                            except:
                                continue
                    elif search_mode == 'string':
                        pred_string = self.tokenizer.decode(pred_token_ids)
                        idx = search(pred_string, text)
                        if idx != -1:
                            pred_tuple.append((schema_type, role, pred_string))
                    
            schema_output_list.append(pred_tuple)
        return schema_output_list
    
    
class single_schema_type_decoder(object):
    def __init__(self, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train'):
        
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
                
        
        self.trie_tree = KeywordProcessor()
        for token in [self.sep_token, self.seg_token, self.group_token, self.start_token, self.end_token]:
            tokens = self.tokenizer.tokenize(token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(tokens), token)
            
        for i in range(1, 100):
            sentinel_token = self.sentinel_token.format(i)
            sentinel_tokens = self.tokenizer.tokenize(sentinel_token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(sentinel_tokens), sentinel_token)
            
    def single_schema_decode(self, text, token2char_span_mapping, schema_type, result_dict, query_token_ids, ori_input_ids, mode='unilm', search_mode='token_id'):
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        
        if isinstance(result_dict.sequences, torch.Tensor):
            token_ids = result_dict.sequences[0].cpu().numpy().tolist()
        else:
            token_ids = result_dict.sequences[0]
            
        # print(self.tokenizer.decode(token_ids))
        
        if mode == 'unilm':
            input_len = len(query_token_ids)
            generated_token_ids = token_ids[input_len:]
        elif mode == 'seq2seq_no_decoder_input':
            generated_token_ids = token_ids[1:]
            
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        schema_dict = self.schema_dict
        
        group_output_list = []
        start_index = 0
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break
                
            if word in [self.seg_token]:
                generated_token_ids = generated_token_ids[end:]
                
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        group_output_list = []
        start_index = 0
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break
    
            pred_tokens = []
            group_tokens = []

            if word in [self.group_token]:
                group_token_ids = generated_token_ids[start_index:start] # value-tokens + tole_type-tokens
                start_index = end
                group_output_list.append(group_token_ids)
            
        schema_output_list = []
        for group_token_ids in group_output_list:
            keywords = self.trie_tree.extract_keywords(group_token_ids, span_info=True)
            start_index = 0
            pred_tuple = []
            for keyword in keywords:
                word = keyword[0]
                start = keyword[1]
                end = keyword[2]
                if word in schema_dict['sentinel2role']:
                    role_type = schema_dict['sentinel2role'][word]
                    role = schema_dict['type2role'][role_type]
                    pred_token_ids = group_token_ids[start_index:start] # value-tokens + tole_type-tokens
                    start_index = end
                    schema_output_list.append((self.tokenizer.decode(pred_token_ids), ))
        return schema_output_list

class single_schema_decoder_no_match(object):
    def __init__(self, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train'):
        
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
                
        
        self.trie_tree = KeywordProcessor()
        for token in [self.sep_token, self.seg_token, self.group_token, self.start_token, self.end_token]:
            tokens = self.tokenizer.tokenize(token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(tokens), token)
            
        for i in range(1, 100):
            sentinel_token = self.sentinel_token.format(i)
            sentinel_tokens = self.tokenizer.tokenize(sentinel_token)
            self.trie_tree.add_keyword(self.tokenizer.convert_tokens_to_ids(sentinel_tokens), sentinel_token)
            
    def single_schema_decode(self, text, token2char_span_mapping, schema_type, result_dict, query_token_ids, ori_input_ids, mode='unilm', search_mode='token_id'):
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        
        if isinstance(result_dict.sequences, torch.Tensor):
            token_ids = result_dict.sequences[0].cpu().numpy().tolist()
        else:
            token_ids = result_dict.sequences[0]
            
        # print(self.tokenizer.decode(token_ids))
        
        if mode == 'unilm':
            input_len = len(query_token_ids)
            generated_token_ids = token_ids[input_len:]
        elif mode == 'seq2seq_no_decoder_input':
            generated_token_ids = token_ids[1:]
            
        keywords = self.trie_tree.extract_keywords(generated_token_ids, span_info=True)
        schema_dict = self.schema_dict[schema_type]
            
        group_output_list = []
        start_index = 0
        for keyword in keywords:
            word = keyword[0]
            start = keyword[1]
            end = keyword[2]
            if word in [self.end_token]:
                break

            pred_tokens = []
            group_tokens = []

            if word in [self.group_token]:
                group_token_ids = generated_token_ids[start_index:start] # value-tokens + tole_type-tokens
                start_index = end
                group_output_list.append(group_token_ids)
        
        schema_output_list = []
        for group_token_ids in group_output_list:
            keywords = self.trie_tree.extract_keywords(group_token_ids, span_info=True)
            start_index = 0
            pred_tuple = []
            for keyword in keywords:
                word = keyword[0]
                start = keyword[1]
                end = keyword[2]
                if word in schema_dict['sentinel2role']:
                    role_type = schema_dict['sentinel2role'][word]
                    role = schema_dict['type2role'][role_type]
                    pred_token_ids = group_token_ids[start_index:start] # value-tokens + tole_type-tokens
                    start_index = end
                    
                    if search_mode == 'token_id':
                        idx = search(pred_token_ids, ori_input_ids)
                        if idx != -1:
                            try:
                                start, end = new_span[idx][0], new_span[idx+len(pred_token_ids)-1][-1] + 1
                                pred_tuple.append((schema_type, role, text[start:end]))
                            except:
                                continue
                        else:
                            pred_tuple.append((schema_type, role, self.tokenizer.decode(pred_token_ids)))
                    elif search_mode == 'string':
                        pred_string = self.tokenizer.decode(pred_token_ids)
                        idx = search(pred_string, text)
                        if idx != -1:
                            pred_tuple.append((schema_type, role, pred_string))
                        else:
                            pred_tuple.append((schema_type, role, self.tokenizer.decode(pred_token_ids)))
                    
            schema_output_list.append(pred_tuple)
        return schema_output_list