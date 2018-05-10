import argparse
import cPickle
import h5py
import os
import numpy as np

from tqdm import tqdm
from itertools import groupby
from collections import Counter
from nltk.corpus import stopwords  # remove stopwords and too frequent words (in, a, the ..)
from collections import defaultdict

from util import log


def str_list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(',')]

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--enwiki_dirs', type=str_list,
                    default='data/preprocessed/enwiki/enwiki_processed_1_4,' \
                            'data/preprocessed/enwiki/enwiki_processed_2_4,' \
                            'data/preprocessed/enwiki/enwiki_processed_3_4,' \
                            'data/preprocessed/enwiki/enwiki_processed_4_4', help=' ')
parser.add_argument('--genome_annotation_dir', type=str,
                    default='data/VisualGenome/annotations', help=' ')
parser.add_argument('--dir_name', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
parser.add_argument('--context_window_size', type=int, default=3,
                    help='window size for extracting context')
parser.add_argument('--preprocessing', type=int, default=0,
                    help='whether to do preprocessing (1) or not (0)')
parser.add_argument('--min_num_word', type=int, default=5, help='min num word in set')
config = parser.parse_args()

config.answer_dict_path = os.path.join(config.dir_name, 'answer_dict.pkl')
answer_dict = cPickle.load(open(config.answer_dict_path, 'rb'))

word2contexts = None

for enwiki_dir in tqdm(config.enwiki_dirs, desc="merging word2contexts"):
    word2contexts_path = os.path.join(
        enwiki_dir, 'word2contexts_w{}_p{}.pkl'.format(
            config.context_window_size,
            int(config.preprocessing)))

    log.info('loading word2context.. {}'.format(word2contexts_path))
    cur_word2contexts = cPickle.load(open(word2contexts_path, 'rb'))

    if word2contexts is None:
        word2contexts = cur_word2contexts
    else:
        for word, counter in tqdm(cur_word2contexts.items()):
            for context, count in counter.items():
                word2contexts[word][context] += count

log.info('word2contexts done')

context2word_list = {}
for v in tqdm(word2contexts, desc='build context2word_list'):
    for context in word2contexts[v]:
        if context not in context2word_list:
            context2word_list[context] = []
        context2word_list[context].append(v)

new_context2word_list = {}
for context in tqdm(context2word_list, desc='filter wordset contexts'):
    if len(context2word_list[context]) >= config.min_num_word:
        new_context2word_list[context] = context2word_list[context]

wordlist_with_cnt = []
for context in tqdm(new_context2word_list, desc='wordlist_with_cnt'):
    word_list = new_context2word_list[context]
    count_sum = sum([word2contexts[w][context] for w in word_list])
    wordlist_with_cnt.append((word_list, context, count_sum))

sorted_wordlist_with_cnt_by_t = sorted(
    wordlist_with_cnt,
    key=lambda x: len(x[1].split()) - 1 - x[1].split().count('<unk>'),
    reverse=True)

filtered_sorted_wordlist = [k for k in sorted_wordlist_with_cnt_by_t
                            if '<unk> <word> <unk>' not in k[1]]

reduced_sorted_wordlist = [w for w in filtered_sorted_wordlist
                           if len(w[1].split()) - 1 - w[1].split().count('<unk>') >= 2]
reduced_sorted_wordlist.append((answer_dict['vocab'], '<word>', 1))  # default context
context_list = [w[1] for w in reduced_sorted_wordlist]
context_vocab = set()
for context in context_list:
    for w in context.split():
        context_vocab.add(w)
context_vocab = list(context_vocab)
context_vocab_dict = {w: i for i, w in enumerate(context_vocab)}
context2idx = {context: idx for idx, context in enumerate(context_list)}
context2weight = {context: (len(context.split()) - 1 - context.split().count('<unk>'))**2
                  for context in context_list}
context2weight['<word>'] = 1  # default context
max_context_len = max([len(context.split()) for context in context_list])
np_context = np.zeros([len(context_list), max_context_len], dtype=np.int32)
np_context_len = np.zeros([len(context_list)], dtype=np.int32)
for context in tqdm(context_list, desc='np_context'):
    context_tokens = context.split()
    context_intseq = [context_vocab_dict[t] for t in context_tokens]
    context_intseq_len = len(context_intseq)
    context_idx = context2idx[context]
    np_context[context_idx, :context_intseq_len] = context_intseq
    np_context_len[context_idx] = context_intseq_len

ans2context_idx = {}
for context_tuple in reduced_sorted_wordlist:
    context_idx = context2idx[context_tuple[1]]
    for ans in context_tuple[0]:
        ans_idx = answer_dict['dict'][ans]
        if ans_idx not in ans2context_idx:
            ans2context_idx[ans_idx] = []
        ans2context_idx[ans_idx].append(context_idx)

ans2context_prob = {}
for ans in tqdm(ans2context_idx, desc='ans2context_prob'):
    ans2context_prob[ans] = []
    for context_idx in ans2context_idx[ans]:
        weight = context2weight[context_list[context_idx]]
        ans2context_prob[ans].append(weight)
    partition = float(sum(ans2context_prob[ans]))
    ans2context_prob[ans] = [w / partition for w in ans2context_prob[ans]]

enwiki_context_dict = {
    'idx2context': context_list,
    'context2idx': context2idx,
    'context2weight': context2weight,
    'max_context_len': max_context_len,
    'context_word_vocab': context_vocab,
    'context_word_dict': context_vocab_dict,
    'ans2context_idx': ans2context_idx,
    'ans2context_prob': ans2context_prob,
}
save_name = 'enwiki_context_dict_w{}_p{}_n{}'.format(
    config.context_window_size, config.preprocessing, config.min_num_word)
save_pkl_path = os.path.join(config.dir_name, '{}.pkl'.format(save_name))
log.info('saving: {} ..'.format(save_pkl_path))
cPickle.dump(enwiki_context_dict, open(save_pkl_path, 'wb'))

save_h5_path = os.path.join(config.dir_name, '{}.hdf5'.format(save_name))
log.info('saving: {} ..'.format(save_h5_path))
with h5py.File(save_h5_path, 'w') as f:
    f['np_context'] = np_context
    f['np_context_len'] = np_context_len
log.warn('done')
