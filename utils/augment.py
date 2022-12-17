

# AEDA: An Easier Data Augmentation Technique for Text classification
# Akbar Karimi, Leonardo Rossi, Andrea Prati

import random

random.seed(0)

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':', '。', '，', '！', '？', '；', '：', '【', '】', '（', '）', '(', ')']
NUM_AUGS = [1, 2, 4, 8]
PUNC_RATIO = 0.1

# Insert punction words into a given sentence with the given ratio "punc_ratio"
def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
    words = sentence.split()
    new_line = []
    q = random.randint(1, int(punc_ratio * len(sentence) + 1))
    qs = random.sample(range(0, len(words)), q)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)
    return new_line