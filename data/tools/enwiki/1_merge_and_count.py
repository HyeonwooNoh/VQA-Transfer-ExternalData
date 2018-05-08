import argparse
import cPickle
import glob
import os

from collections import Counter
from tqdm import tqdm

from util import log

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--enwiki_dir', type=str,
                    default='data/preprocessed/enwiki/enwiki_processed', help=' ')
parser.add_argument('--genome_annotation_dir', type=str,
                    default='data/VisualGenome/annotations', help=' ')
parser.add_argument('--dir_name', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
config = parser.parse_args()

config.enwiki_paths = glob.glob(os.path.join(config.enwiki_dir, 'wiki_*'))

config.answer_dict_path = os.path.join(config.dir_name, 'answer_dict.pkl')
answer_dict = cPickle.load(open(config.answer_dict_path, 'rb'))

vocab_1st_token_set = set([v.split()[0] for v in answer_dict['vocab']])
vocab_1st2sent_idx_list = {v: [] for v in list(vocab_1st_token_set)}

word_list = []
used_sents = []
cur_sent_idx = 0
for i, enwiki_path in enumerate(config.enwiki_paths):
    log.infov('loading enwiki [{}/{}]: {}'.format(
        i, len(config.enwiki_paths), enwiki_path))

    sent_list = open(enwiki_path, 'r').read().splitlines()

    for sent in tqdm(sent_list, desc='process sents'):
        sent_tokens = sent.split()
        overlap = set(sent_tokens) & vocab_1st_token_set

        if len(overlap) > 0:
            for v in list(overlap):
                vocab_1st2sent_idx_list[v].append(cur_sent_idx)
            cur_sent_idx += 1
            used_sents.append(sent)
            word_list.extend(sent_tokens)
log.info('counting words..')
word_cnt = Counter(word_list)
log.info('done')
most_common = word_cnt.most_common()

save_path = os.path.join(config.enwiki_dir, 'merge_and_count.pkl')
log.warn('saving results to : {}'.format(save_path))
cPickle.dump({
    'vocab_1st2sent_idx_list': vocab_1st2sent_idx_list,
    'most_common': most_common,
}, open(save_path, 'wb'))
save_text_path = os.path.join(config.enwiki_dir, 'merged_used_sents.txt')
log.warn('saving results to : {}'.format(save_text_path))
f = open(save_text_path, 'w')
for sent in tqdm(used_sents, desc='saving texts'):
    f.write(sent + '\n')
log.warn('done')
