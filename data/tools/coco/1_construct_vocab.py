import argparse
import collections
import cPickle
import json
import os

from tqdm import tqdm

from util import log

DEBUG = False
GLOVE_VOCAB_PATH = 'data/preprocessed/glove_vocab.json'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--caption_split_dir', type=str,
                    default='data/preprocessed/COCO/standrad', help=' ')
parser.add_argument('--caption_dic_path', type=str,
                    default='data/COCO/dic_coco.json', help=' ')
parser.add_argument('--caption_cap_path', type=str,
                    default='data/COCO/cap_coco.json', help=' ')
parser.add_argument('--reference_vqa_dir', type=str,
                    default='data/preprocessed/vqa_v2/'
                    'qa_split_objattr_answer_genome_memft_check_all_answer_thres1_50000_thres2_-1')
parser.add_argument('--answer_set_limit', type=int, default=3000, help=' ')
parser.add_argument('--max_answer_len', type=int, default=3, help=' ')
config = parser.parse_args()


###################################
# get preprocessed data from vqa
###################################

dirname, filename = os.path.split(os.path.abspath(__file__))
root_dir = "/".join(dirname.split("/")[:-3])

def symlink(filename):
    src_path = os.path.join(root_dir, config.reference_vqa_dir, filename)
    dst_path = os.path.join(root_dir, config.caption_split_dir, filename)

    if os.path.exists(dst_path):
        log.info("{} already exists".format(dst_path))
    else:
        log.info("Sym link: {}->{}".format(src_path, dst_path))
        os.symlink(src_path, dst_path)

symlink("object_list.pkl")
symlink("attribute_list.pkl")
symlink("obj_attrs_split.pkl")

caption_dic = json.load(open(config.caption_dic_path))
caption_cap = json.load(open(config.caption_cap_path))

def makedirs(path):
    path = str(path)
    if not os.path.exists(path):
        log.info("Make directories: {}".format(path))
        os.makedirs(path)
    else:
        log.info("Skip making directories: {}".format(path))

makedirs(config.caption_split_dir)

caption_split = collections.defaultdict(list)
for info in caption_dic['images']:
    caption_split[info['split']].append(info['id'])

caption_split_path = os.path.join(
    config.caption_split_dir, "caption_split.json")

with open(caption_split_path, 'w') as f:
    json.dump(caption_split, f)

# {'test': xxx, 'train': xxx }
id2caption = {}
caption_idxs = collections.defaultdict(list) 

# separate out indexes for each of the provided splits
for key in caption_split:
    for ix in range(len(caption_dic['images'])):
        img = caption_dic['images'][ix]
        if img['split'] == key:
            caption_idxs[key].append(ix)
            ix2caption[int(img['id'])] = caption_cap[ix]

ix2caption_path = os.path.join(
    config.ix2caption_dir, "ix2caption.json")
with open(ix2caption_path, 'w') as f:
    json.dump(caption_idxs, f)


for key in caption_split:
    log.info("# of {}: {}".format(key, len(caption_split[key])))

noc_object = [
        'bus', 'bottle', 'couch', 'microwave',
        'pizza', 'racket', 'suitcase', 'zebra']
noc_index = [6, 40, 58, 69, 54, 39, 29, 23]

noc_word_map = {
        'bus': ['bus', 'busses'],
        'bottle': ['bottle', 'bottles'],
        'couch': ['couch', 'couches', 'sofa', 'sofas'],
        'microwave': ['microwave', 'microwaves'],
        'pizza': ['pizza', 'pizzas'],
        'racket': ['racket', 'rackets', 'racquet', 'racquets'],
        'suitcase': ['luggage', 'luggages', 'suitcase', 'suitcases'],
        'zebra': ['zebra', 'zebras']
}
noc_words = [word for words in noc_word_map.values() for word in words]

# check whether noc words in captions
if DEBUG:
    for ix in caption_idxs['train']:
        captions = caption_cap[ix]
        image_id = caption_dic['images'][ix]['id']
        file_path = caption_dic['images'][ix]['file_path']

        if any((word in noc_words) for cap in captions for word in cap):
            print("test", captions)

#################
# Build vocab
#################

# below may not be used
#itow = caption_dic['ix_to_word']
#wtoi = {w:i for i,w in itow.items()}
## word to detection
#wtod = {w:i+1 for w,i in caption_dic['wtod'].items()}
#dtoi = {w:i+1 for i,w in enumerate(wtod.keys())} # detection to index
#itod = {i+1:w for i,w in enumerate(wtod.keys())}
#wtol = caption_dic['wtol']
#ltow = {l:w for w,l in wtol.items()}
#vocab_size = len(itow) + 1 # since it start from 1
#print('vocab size is ', vocab_size)

log.info('loading glove_vocab..')
glove_vocab = json.load(open(GLOVE_VOCAB_PATH, 'r'))
log.info('done')

obj_attrs_split_path = os.path.join(
        config.caption_split_dir, 'obj_attrs_split.pkl')
obj_attrs_split = cPickle.load(open(obj_attrs_split_path, 'rb'))

"""
Filtering caption:
    - count answer occurrences and filter rare answers
      (filtering answer candidates is for efficiency of answer classification)
    - make answer set with only frequent answers
    - make sure that every chosen answers consist of glove vocabs
    - all questions are used for vocab construction
*: When making GT, let's make N(answer) + 1 slots and if answer is not in answer
set, mark gt to N(answer) + 1 slot (only for testing data). For training data,
we will just ignore caption with rare answers.
"""

glove_vocab_set = set(glove_vocab['vocab'])

answers = list()
for captions in tqdm(caption_cap, desc='count answers'):
    for tokens in captions:
        answers.append(' '.join(tokens))
answer_counts = collections.Counter(answers)

ans_in_order = list(zip(*answer_counts.most_common())[0])
ans_in_order_glove = []
for ans in ans_in_order:
    in_glove = True
    a_tokens = ans.split()
    if len(a_tokens) > config.max_answer_len:
        continue
    for t in a_tokens:
        if t not in glove_vocab_set:
            in_glove = False
            break
    if in_glove:
        ans_in_order_glove.append(ans)

freq_ans = ans_in_order_glove[:config.answer_set_limit]
freq_ans_set = set(freq_ans)

q_vocab = set()
for captions in tqdm(caption_cap, desc='count q_vocab'):
    for tokens in captions:
        for t in tokens: q_vocab.add(t)
q_vocab = q_vocab & glove_vocab_set

a_vocab = set()
for ans in tqdm(freq_ans, desc='count a_vocab'):
    for t in ans.split(): a_vocab.add(t)
if len(a_vocab - glove_vocab_set) > 0:
    raise RuntimeError('Somethings wrong: a_vocab is already filtered')
caption_vocab = q_vocab | a_vocab
caption_vocab = list(caption_vocab)
caption_vocab.append('<s>')
caption_vocab.append('<e>')
caption_vocab.append('<unk>')

"""
How to store vocab:
    - ['vocab'] = ['yes', 'no', 'apple', ...]
    - ['dict'] = {'yes': 0, 'no: 1, 'apple': 2, ...}
    - in a json format
"""
save_vocab = {}
save_vocab['vocab'] = caption_vocab
save_vocab['dict'] = {v: i for i, v in enumerate(caption_vocab)}

vocab_path = os.path.join(config.caption_split_dir, 'vocab.pkl')
log.warn('save vocab: {}'.format(vocab_path))
cPickle.dump(save_vocab, open(vocab_path, 'wb'))

vocab_path = os.path.join(config.caption_split_dir, 'vocab.json')
log.warn('save vocab: {}'.format(vocab_path))
json.dump(save_vocab, open(vocab_path, 'w'))

test_object_set = set(obj_attrs_split['test'])
train_ans_set = freq_ans_set - test_object_set
test_ans_set = freq_ans_set & test_object_set

log.info('loading object and attribute list')
obj_list_path = os.path.join(config.caption_split_dir, 'object_list.pkl')
obj_list = cPickle.load(open(obj_list_path, 'rb'))
attr_list_path = os.path.join(config.caption_split_dir, 'attribute_list.pkl')
attr_list = cPickle.load(open(attr_list_path, 'rb'))
obj_set = set(obj_list)
attr_set = set(attr_list)

answer_dict = {}
answer_dict['vocab'] = list(train_ans_set) + list(test_ans_set)
answer_dict['num_train_answer'] = len(train_ans_set)
answer_dict['dict'] = {v: i for i, v in enumerate(answer_dict['vocab'])}
answer_dict['is_object'] = [int(v in obj_set) for v in answer_dict['vocab']]
answer_dict['is_attribute'] = [int(v in attr_set) for v in answer_dict['vocab']]

answer_dict_path = os.path.join(config.caption_split_dir, 'answer_dict.pkl')
log.warn('save answer_dict: {}'.format(answer_dict_path))
cPickle.dump(answer_dict, open(answer_dict_path, 'wb'))

freq_ans_path = os.path.join(
    config.caption_split_dir, 'frequent_answers.json')
log.warn('save frequent answers: {}'.format(freq_ans_path))
json.dump(freq_ans, open(freq_ans_path, 'w'))

log.warn('done')
