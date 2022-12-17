# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
https://github.com/facebookresearch/GENRE/blob/main/examples_genre/examples.ipynb
"""

from typing import Dict, List

try:
    import marisa_trie
except ModuleNotFoundError:
    pass


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.trie_dict, self.append_trie, self.bos_token_id, 'sequence'
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        ori_trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
        mode='sequence'
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie:
                if mode == 'sequence':
                    output += list(append_trie.trie_dict.keys())
                elif mode == 'append' and len(output) == 0:
                    output += list(set(list(ori_trie_dict.keys()))|set(list(append_trie.trie_dict.keys())))
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                ori_trie_dict,
                append_trie,
                bos_token_id,
                mode=mode
            )
        elif prefix_sequence[0] in append_trie.trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                append_trie.trie_dict[prefix_sequence[0]],
                ori_trie_dict,
                append_trie,
                bos_token_id,
                mode='append'
            )
        elif prefix_sequence[0] in ori_trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                ori_trie_dict[prefix_sequence[0]],
                ori_trie_dict,
                append_trie,
                bos_token_id,
                mode='sequence'
            )
        else:
            if append_trie:
                return list(set(list(ori_trie_dict.keys()))|set(list(append_trie.trie_dict.keys())))
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)

# class Trie(object):
#     def __init__(self, sequences: List[List[int]] = []):
#         self.trie_dict = {}
#         self.len = 0
#         if sequences:
#             for sequence in sequences:
#                 Trie._add_to_trie(sequence, self.trie_dict)
#                 self.len += 1

#         self.append_trie = None
#         self.bos_token_id = None

#     def append(self, trie, bos_token_id):
#         self.append_trie = trie
#         self.bos_token_id = bos_token_id

#     def add(self, sequence: List[int]):
#         Trie._add_to_trie(sequence, self.trie_dict)
#         self.len += 1

#     def get(self, prefix_sequence: List[int]):
#         return Trie._get_from_trie(
#             prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
#         )

#     @staticmethod
#     def load_from_dict(trie_dict):
#         trie = Trie()
#         trie.trie_dict = trie_dict
#         trie.len = sum(1 for _ in trie)
#         return trie

#     @staticmethod
#     def _add_to_trie(sequence: List[int], trie_dict: Dict):
#         if sequence:
#             if sequence[0] not in trie_dict:
#                 trie_dict[sequence[0]] = {}
#             Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

#     @staticmethod
#     def _get_from_trie(
#         prefix_sequence: List[int],
#         trie_dict: Dict,
#         append_trie=None,
#         bos_token_id: int = None,
#     ):
#         if len(prefix_sequence) == 0:
#             output = list(trie_dict.keys())
#             if append_trie and bos_token_id in output:
#                 output.remove(bos_token_id)
#                 output += list(append_trie.trie_dict.keys())
#             return output
#         elif prefix_sequence[0] in trie_dict:
#             return Trie._get_from_trie(
#                 prefix_sequence[1:],
#                 trie_dict[prefix_sequence[0]],
#                 append_trie,
#                 bos_token_id,
#             )
#         else:
#             if append_trie:
#                 return append_trie.get(prefix_sequence)
#             else:
#                 return []

#     def __iter__(self):
#         def _traverse(prefix_sequence, trie_dict):
#             if trie_dict:
#                 for next_token in trie_dict:
#                     yield from _traverse(
#                         prefix_sequence + [next_token], trie_dict[next_token]
#                     )
#             else:
#                 yield prefix_sequence

#         return _traverse([], self.trie_dict)

#     def __len__(self):
#         return self.len

#     def __getitem__(self, value):
#         return self.get(value)

        
def flat_list(h_list):
    e_list = []

    for item in h_list:
        if isinstance(item, list):
            e_list.extend(flat_list(item))
        else:
            e_list.append(item)
    return e_list
        
def get_end_to_end_prefix_allowed_tokens_fn_hf(input_ids, task_dict, tokenizer):
    sentinel_token = task_dict['sentinel_token']
    sep_token = task_dict['sep_token']
    seg_token = task_dict['seg_token']
    group_token = task_dict['group_token']
    start_token = task_dict['start_token']
    end_token = task_dict['end_token']
    
    tokenizer.add_special_tokens({
                    "additional_special_tokens": [ sentinel_token.format(i) for i in range(1, 100)]+[seg_token, sep_token, 
                                                                                                group_token, start_token, end_token] })
    
    candies_all = flat_list([[(i, j) for j in range(i, min(20+i, len(input_ids)))] for i in range(0, len(input_ids))])
    all_candidies = [input_ids[i:j+1] for (i,j) in candies_all]

    sentinel_id_sequences = [tokenizer(sentinel_token.format(i), add_special_tokens=False)['input_ids'] for i in range(1,100)]
    sentinel_id_sequences += [tokenizer(item, add_special_tokens=False)['input_ids'] for item in [seg_token, sep_token, 
                                                                                                group_token, start_token, end_token]]
    
    trie = Trie(all_candidies)
    append_trie = Trie(sentinel_id_sequences)
    trie.append(append_trie, 0)
        
    def prefix_allowed_tokens_fn(batch_id, sent):
        return trie.get(sent)
    
    return prefix_allowed_tokens_fn