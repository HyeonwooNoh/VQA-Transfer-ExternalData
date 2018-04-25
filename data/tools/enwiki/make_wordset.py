import argparse
import cPickle
import glob
import json
import os

from collections import Counter
from itertools import groupby
from nltk.corpus import stopwords  # remove stopwords and too frequent words (in, a, the ..)
from tqdm import tqdm

from util import log

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--enwiki_dir', type=str,
                    default='data/preprocessed/enwiki/enwiki_processed_backup', help=' ')
parser.add_argument('--genome_annotation_dir', type=str,
                    default='data/VisualGenome/annotations', help=' ')
parser.add_argument('--dir_name', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
parser.add_argument('--context_window_size', type=int, default=3,
                    help='window size for extracting context')
parser.add_argument('--min_num_word', type=int, default=5, help='min num word in set')
config = parser.parse_args()

config.answer_dict_path = os.path.join(config.dir_name, 'answer_dict.pkl')
answer_dict = cPickle.load(open(config.answer_dict_path, 'rb'))

log.info('loading word2context..')
word2contexts = cPickle.load(open(
    os.path.join(config.enwiki_dir, 'word2contexts.pkl'), 'rb'))
log.info('done')

context2word_list = {}
for v in tqdm(word2contexts):
    for context in word2contexts[v]:
        if context not in context2word_list:
            context2word_list[context] = []
        context2word_list[context].append(v)
