import argparse
import cPickle
import copy
import os

from collections import Counter
from itertools import groupby
from nltk.corpus import stopwords  # remove stopwords and too frequent words (in, a, the ..)
from tqdm import tqdm

from util import log

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--bookcorpus_dir', type=str,
                    default='data/preprocessed/bookcorpus/bookcorpus_processed', help=' ')
parser.add_argument('--genome_annotation_dir', type=str,
                    default='data/VisualGenome/annotations', help=' ')
parser.add_argument('--dir_name', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
parser.add_argument('--context_window_size', type=int, default=3,
                    help='window size for extracting context')
config = parser.parse_args()

config.answer_dict_path = os.path.join(config.dir_name, 'answer_dict.pkl')
answer_dict = cPickle.load(open(config.answer_dict_path, 'rb'))

vocab_1st_token_set = set([v.split()[0] for v in answer_dict['vocab']])

log.info('loading merge_and_count..')
merge_and_count = cPickle.load(open(
    os.path.join(config.bookcorpus_dir, 'merge_and_count.pkl'), 'rb'))
vocab_1st2sent_idx_list = merge_and_count['vocab_1st2sent_idx_list']
most_common = merge_and_count['most_common']

log.info('loading merged_used_sents..')
used_sents = open(os.path.join(
    config.bookcorpus_dir, 'merged_used_sents.txt'), 'r').read().splitlines()
bookcorpus_tokens = [sent.split() for sent in tqdm(used_sents, desc='tokenize sents')]
log.info('done')

freq_word_set = set(zip(*most_common[:100])[0])
stopWords = set(stopwords.words('english'))

log.warn('number of tokenized sentences: {}'.format(len(bookcorpus_tokens)))

t_unk = '<unk>'
t_word = '<word>'
w_sz = config.context_window_size
word2contexts = {}
for i, v in enumerate(answer_dict['vocab']):
    log.infov('processing vocab {} [{}/{}]'.format(
        v, i, len(answer_dict['vocab'])))

    v_tokens = v.split()
    sent_idx_list = vocab_1st2sent_idx_list[v_tokens[0]]
    v_contexts = []
    for sent_idx in tqdm(sent_idx_list, desc='sent_idx_list: {}'.format(v)):
        sent_tokens = bookcorpus_tokens[sent_idx]
        idx_list = [i for i, t in enumerate(sent_tokens) if t == v_tokens[0]]
        if len(idx_list) == 0: continue
        for idx in idx_list:
            if sent_tokens[idx: idx + len(v_tokens)] != v_tokens: continue
            if len(v_tokens) > 1:
                this_sent_tokens = copy.deepcopy(sent_tokens)
                this_sent_tokens[idx] = ' '.join(this_sent_tokens[idx: idx + len(v_tokens)])
                this_sent_tokens[idx + 1:] = this_sent_tokens[idx + len(v_tokens):]
            else:
                this_sent_tokens = sent_tokens
            context = []
            context += [t_unk] * max(w_sz - idx, 0)
            context += this_sent_tokens[max(idx - w_sz, 0):idx + w_sz + 1]
            context += [t_unk] * max(idx + w_sz + 1 - len(this_sent_tokens), 0)
            v_contexts.append(context)

    for l in range(w_sz * 2 + 1):
        if l == w_sz:
            for context in v_contexts:
                context[l] = t_word
            continue

        t_list = [context[l] for context in v_contexts]
        t_cnt = Counter(t_list)
        t_set = set([t for t in t_cnt if t_cnt[t] > 2])
        for context in v_contexts:
            if context[l] not in t_set: context[l] = t_unk
            if context[l] in freq_word_set: context[l] = t_unk
            if context[l] in stopWords: context[l] = t_unk

    new_v_contexts = []
    for context in v_contexts:
        suppressed = [x[0] for x in groupby(context)]
        if suppressed[0] == t_unk: suppressed = suppressed[1:]
        if suppressed[-1] == t_unk: suppressed = suppressed[:-1]
        if len(suppressed) == 1: continue
        new_v_contexts.append(' '.join(suppressed))

    word2contexts[v] = dict(Counter(new_v_contexts))

save_path = os.path.join(config.bookcorpus_dir, 'word2contexts_w{}.pkl'.format(
    config.context_window_size))
log.warn('saving results to to : {}'.format(save_path))
cPickle.dump(word2contexts, open(save_path, 'wb'))
log.warn('done')
