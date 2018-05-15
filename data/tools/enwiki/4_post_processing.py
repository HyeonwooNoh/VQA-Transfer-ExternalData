import copy
import argparse
import cPickle
import h5py
import os
import numpy as np

import math
from tqdm import tqdm
import multiprocessing
from threading import Thread
from itertools import groupby
from collections import Counter
from nltk.corpus import stopwords  # remove stopwords and too frequent words (in, a, the ..)
from collections import defaultdict, Counter

from util import log


cpu_count = multiprocessing.cpu_count()
num_thread = max(cpu_count - 2, 1)

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

#config.answer_dict_path = os.path.join(config.dir_name, 'answer_dict.pkl')
#answer_dict = cPickle.load(open(config.answer_dict_path, 'rb'))

save_name = 'enwiki_context_dict_w{}_p{}_n{}'.format(
    config.context_window_size, config.preprocessing, config.min_num_word)
save_pkl_path = os.path.join(config.dir_name, '{}.pkl'.format(save_name))


log.info('Reading: {} ..'.format(save_pkl_path))
enwiki_context_dict = cPickle.load(open(save_pkl_path, 'rb'))

if 'ans2shuffled_context_idx' not in enwiki_context_dict:
    log.info("Generating `enwiki_context_dict`...")

    ans2shuffled_context_idx = copy.deepcopy(enwiki_context_dict['ans2context_idx'])
    for ans in ans2shuffled_context_idx:
        np.random.shuffle(ans2shuffled_context_idx[ans])
    enwiki_context_dict['ans2shuffled_context_idx'] = ans2shuffled_context_idx

    cPickle.dump(enwiki_context_dict, open(save_pkl_path, 'wb'))
else:
    log.info("`enwiki_context_dict` already exists. Skip")
