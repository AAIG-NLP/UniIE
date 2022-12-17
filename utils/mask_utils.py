
import numpy as np

class ParagraphInfo(object):
    def __init__(self, vocab):
        self.vocab2id = {}
        self.id2vocab = {}
        for index, word in enumerate(vocab):
            self.vocab2id[word] = index
            self.id2vocab[index] = word

    def is_start_word(self, idx):
        if isinstance(idx, int):
            return not self.id2vocab[idx].startswith('##')
        elif isinstance(idx, str):
            idx = self.vocab2id[idx]
            return not self.id2vocab[idx].startswith('##')

    # def get_word_piece_map(self, sentence):
    #   return [self.is_start_word(i) for i in sentence]

    def get_word_piece_map(self, sentence):
        """
        sentence: word id of sentence,
        [[0,1], [7,10]]
        """
        word_piece_map = []
        for segment in sentence:
            if isinstance(segment, list):
                for index, idx in enumerate(segment):
                    if index == 0 or is_start_word(idx):
                        word_piece_map.append(True)
                    else:
                        word_piece_map.append(self.is_start_word(idx))
            else:
                word_piece_map.append(self.is_start_word(segment))
        return word_piece_map

    def get_word_at_k(self, sentence, left, right, k, word_piece_map=None):
        num_words = 0
        while num_words < k and right < len(sentence):
            # complete current word
            left = right
            right = self.get_word_end(sentence, right, word_piece_map)
            num_words += 1
        return left, right

    def get_word_start(self, sentence, anchor, word_piece_map=None):
        word_piece_map = word_piece_map if word_piece_map is not None else self.get_word_piece_map(sentence)
        left  = anchor
        while left > 0 and word_piece_map[left] == False:
                left -= 1
        return left
    # word end is next word start
    def get_word_end(self, sentence, anchor, word_piece_map=None):
        word_piece_map = word_piece_map if word_piece_map is not None else self.get_word_piece_map(sentence)
        right = anchor + 1
        while right < len(sentence) and word_piece_map[right] == False:
                right += 1
        return right

def pad_to_max(pair_targets, pad):
    max_pair_target_len = max([len(pair_tgt) for pair_tgt in pair_targets])
    for pair_tgt in pair_targets:
        this_len = len(pair_tgt)
        for i in range(this_len, max_pair_target_len):
            pair_tgt.append(pad)
    return pair_targets

def pad_to_len(pair_targets, pad, max_pair_target_len):
    for i in range(len(pair_targets)):
        pair_targets[i] = pair_targets[i][:max_pair_target_len]
        this_len = len(pair_targets[i])
        for j in range(max_pair_target_len - this_len):
            pair_targets[i].append(pad)
    return pair_targets

def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x : x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            merged.append(interval)
        else:
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged

def bert_masking(sentence, mask, tokens, pad, mask_id):
    sentence = np.copy(sentence)
    sent_length = len(sentence)
    target = np.copy(sentence)
    mask = set(mask)
    for i in range(sent_length):
        if i in mask:
            rand = np.random.random()
            if rand < 0.8:
                sentence[i] = mask_id
            elif rand < 0.9:
                # sample random token according to input distribution
                sentence[i] = np.random.choice(tokens)
        else:
            target[i] = pad
    return sentence, target, None

def span_masking(sentence, spans, tokens, pad, mask_id, pad_len, mask, replacement='word_piece', endpoints='external'):
    """
    pair_targets：
    [0:1]: masked start and end pos
    [1:]: masked word
    """
    sentence = np.copy(sentence)
    sent_length = len(sentence)
    target = np.full(sent_length, pad)
    pair_targets = []
    spans = merge_intervals(spans)
    assert len(mask) == sum([e - s + 1 for s,e in spans])
    # print(list(enumerate(sentence)))
    for start, end in spans:
        lower_limit = 0 if endpoints == 'external' else -1
        upper_limit = sent_length - 1 if endpoints == 'external' else sent_length
        if start > lower_limit and end < upper_limit:
            if endpoints == 'external':
                pair_targets += [[start - 1, end + 1]]
            else:
                pair_targets += [[start, end]]
            pair_targets[-1] += [sentence[i] for i in range(start, end + 1)]
        rand = np.random.random()
        for i in range(start, end + 1):
            assert i in mask
            target[i] = sentence[i]
            if replacement == 'word_piece':
                rand = np.random.random()
            if rand < 0.8:
                sentence[i] = mask_id
            elif rand < 0.9:
                # sample random token according to input distribution
                sentence[i] = np.random.choice(tokens)
    pair_targets = pad_to_len(pair_targets, pad, pad_len + 2)
    # if pair_targets is None:
    return sentence, target, pair_targets

def generate_target_ids(input_ids, mask_indexes, mask_tokens):
    """This function takes a list of sentences and generates the pair (input_ids, target_ids) for pretraining the
    model. It implements in a simple way the final T5 denoising objective, as per HuggingFace documentation.
    :param mask_prob: Probability of masking a token.
    :param input_ids: A list of sublists, where the sublists are sequences of input ids (tokenized sentences). This
        mutable sublists are modified within this function, masking the tokens that the model has to denoise for
        pretraining.
    :return: The correspondent target sequences of ids for each input sentence, with the unmasked tokens.
    """
    target_ids = []
    input_ids = np.copy(input_ids)
    mask = [(i in masked_indexes)  # this is True or False
                for i in range(len(input_ids))]
    
    i = 0
    end = len(input_ids)
    masked_spans_counter = 0
    while i < end:
        if mask[i]:
            current_words_masked = [input_ids[i]]
            input_ids[i] = mask_tokens[masked_spans_counter]
            masked_spans_counter += 1
            while i + 1 < end and mask[i + 1]:
                current_words_masked.append(input_ids[i + 1])
                del input_ids[i + 1]
                del mask[i + 1]
                end -= 1
            target_ids.extend(current_words_masked)
        else:
            if len(target_ids) == 0 or target_ids[-1] != mask_tokens[masked_spans_counter]:
                target_ids.append(mask_tokens[masked_spans_counter])
        i += 1
    return input_ids, target_ids


import numpy as np

"""
https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
"""

def create_sentinel_ids(mask_indices, mode='unilm'):
    """
    Sentinel ids creation given the indices that should be masked.
    The start indices of each mask are replaced by the sentinel ids in increasing
    order. Consecutive mask indices to be deleted are replaced with `-1`.
    """
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
    if mode == 't5':
        sentinel_ids = np.where(sentinel_ids != 0, (len(tokenizer) - sentinel_ids), 0)
    elif mode == 'unilm':
        sentinel_ids = np.where(sentinel_ids != 0, (sentinel_ids), 0)
    sentinel_ids -= mask_indices - start_indices
    return sentinel_ids

def filter_input_ids(input_ids, sentinel_ids):
    """
    Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
    This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
    """
    batch_size = input_ids.shape[0]

    input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
    # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
    # masked tokens coming after sentinel tokens and should be removed
    input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
    return input_ids

def filter_target_ids(input_ids, sentinel_ids):
    batch_size = input_ids.shape[0]
    input_ids = input_ids[:, 1:]
    input_ids_final = []
    for input_id in input_ids:
        input_id_ = []
        max_sentinel_id = 0
        for idx in input_id:
            if idx in sentinel_ids:
                input_id_.append(idx-1)
                max_sentinel_id = idx
            else:
                input_id_.append(idx)
        input_id_.append(max_sentinel_id)
        input_ids_final.append(input_id_)
    return input_ids_final
        
def random_spans_noise_mask(length, noise_density, mean_noise_span_length):

    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]