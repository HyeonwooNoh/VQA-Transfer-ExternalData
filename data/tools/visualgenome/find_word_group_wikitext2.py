import argparse
import cPickle
import os

from collections import Counter
from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--genome_annotation_dir', type=str,
                    default='data/VisualGenome/annotations', help=' ')
parser.add_argument('--dir_name', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
config = parser.parse_args()

config.answer_dict_path = os.path.join(
    config.dir_name, 'answer_dict.pkl')

with open('data/wikitext2/train.txt', 'r') as f:
    wikitext2 = []
    for line in f:
        words = line.split() + ['<eos>']
        for word in words:
            wikitext2.append(word.lower())

answer_dict = cPickle.load(open(config.answer_dict_path, 'rb'))


def get_context(idx):
    i = idx
    b = ['<blank>']
    w = wikitext2
    return [
        ' '.join(w[i-2:i] + b + w[i+1:i+3]),
        ' '.join(w[i-2:i] + b + w[i+1:i+2]),
        ' '.join(w[i-1:i] + b + w[i+1:i+3]),
        ' '.join(w[i-2:i] + b),
        ' '.join(w[i-1:i] + b + w[i+1:i+2]),
        ' '.join(b + w[i+1:i+3]),
        ' '.join(w[i-1:i] + b),
        ' '.join(b + w[i+1:i+2]),
    ]

vocab_set = set(answer_dict['vocab'])
idx_list = {}
for v in list(vocab_set):
    idx_list[v] = []
for idx, v in tqdm(enumerate(wikitext2), desc='idx_list'):
    if v in vocab_set:
        idx_list[v].append(idx)

context_dict = {}
for v in tqdm(answer_dict['vocab'], desc='process'):
    context_dict[v] = []
    for idx in idx_list[v]:
        context_dict[v].extend(get_context(idx))
    context_dict[v] = set(context_dict[v])

cPickle.dump(context_dict, open('temp_context_dict.pkl', 'wb'))

context_list = []
for v in tqdm(context_dict.keys()):
    context_list.extend(list(context_dict[v]))

context_count = Counter(context_list)

overlap_context = set()
for c in context_list:
    if context_count[c] > 10:
        overlap_context.add(c)
overlap_context = list(overlap_context)

wordset = {}
for c in tqdm(overlap_context):
    wordset[c] = []
    for v, v_set in context_dict.items():
        if c in v_set:
            wordset[c].append(v)
