import argparse
import h5py
import json
import os
import re
import numpy as np

from tqdm import tqdm


RANDOM_STATE = np.random.RandomState(123)


QUESTION_PATHS = {
    'train': 'data/VQA_v2/questions'
    '/v2_OpenEnded_mscoco_train2014_questions.json',
    'val': 'data/VQA_v2/questions'
    '/v2_OpenEnded_mscoco_val2014_questions.json',
}
ANNOTATION_PATHS = {
    'train': 'data/VQA_v2/annotations'
    '/v2_mscoco_train2014_annotations.json',
    'val': 'data/VQA_v2/annotations'
    '/v2_mscoco_val2014_annotations.json',
}

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vocab_path', type=str,
                    default='data/preprocessed/new_vocab50.json', help=' ')
parser.add_argument('--vlmap_traindata_dir', type=str,
                    default='data/preprocessed/visualgenome'
                    '/merged_by_image_new_vocab50_min_region20', help=' ')
parser.add_argument('--occ_thres_1', type=int, default=500,
                    help='object classes with occurrence greater or equal to '
                    'this threshold are splited into train')
parser.add_argument('--occ_thres_2', type=int, default=50,
                    help='object classes with occurrence greater or equal to '
                    'this threshold are splited into train and test. If '
                    'occurrence is smaller, they are splited into train-reserve '
                    'and test.')
config = parser.parse_args()

vocab = json.load(open(config.vocab_path, 'r'))


def intseq2str(intseq):
    return ' '.join([vocab['vocab'][i] for i in intseq])

with h5py.File(os.path.join(config.vlmap_traindata_dir, 'data.hdf5'), 'r') as f:
    objects_intseq = f['data_info']['objects_intseq'].value
    objects_intseq_len = f['data_info']['objects_intseq_len'].value

objects = []
for intseq, intseq_len in zip(objects_intseq, objects_intseq_len):
    objects.append(intseq2str(intseq[:intseq_len]))

"""
When we split objects, we consider every object classes independent. Therefore,
splits such as "right leg" and "left leg" could happen. This is because making
disjoint set based on the vocabulary is difficult as most vocabulary is
connected to each other and more than 900 vocabulary end up belongs to the same
clusters. For splits including subsets such as "leg" and "right leg", we allocate
questions to longest object name split.
"""
questions = {}
for key, path in tqdm(QUESTION_PATHS.items(), desc='Loading questions'):
    questions[key] = json.load(open(path, 'r'))
merge_questions = []
for key, entry in questions.items():
    merge_questions.extend(entry['questions'])

annotations = {}
for key, path in tqdm(ANNOTATION_PATHS.items(), desc='Loading annotations'):
    annotations[key] = json.load(open(path, 'r'))
merge_annotations = []
for key, entry in annotations.items():
    data_subtype = entry['data_subtype']
    for anno in tqdm(entry['annotations'], desc='Annotation {}'.format(key)):
        anno['image_path'] = '{}/COCO_{}_{:012d}.jpg'.format(
            data_subtype, data_subtype, anno['image_id'])
        anno['split'] = data_subtype
        merge_annotations.append(anno)

qid2anno = {a['question_id']: a for a in merge_annotations}
for q in tqdm(merge_questions, desc='merge question and annotations'):
    qid2anno[q['question_id']]['question'] = q['question']


def split_with_punctuation(string):
    return re.findall(r"[\w']+|[.,!?;]", string)

for qid in tqdm(qid2anno.keys(), desc='tokenize QA':
    anno = qid2anno[qid]
    qid2anno[qid]['q_tokens'] = split_with_punctuation(
        anno['question'].lower())
    qid2anno[qid]['a_tokens'] = split_with_punctuation(
        anno['multiple_choice_answer'].lower())

"""
Test question or answer could have training object words, but it should contain
at least one test object words, which is unseen during training.
"""


def get_ngrams(tokens, n):
    ngrams = []
    for i in range(0, len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i: i+n]))
    return ngrams

occurrence = {name: 0 for name in objects}
max_name_len = objects_intseq_len.max()
for anno in tqdm(qid2anno.values(), desc='count object occurrence'):
    q_tokens = anno['q_tokens']
    a_tokens = anno['a_tokens']
    grams = []
    for n in range(1, min(len(q_tokens), max_name_len)):
        grams.extend(get_ngrams(q_tokens, n))
    for n in range(1, min(len(a_tokens), max_name_len)):
        grams.extend(get_ngrams(a_tokens, n))
    grams = list(set(grams))
    for gram in grams:
        if gram in occurrence: occurrence[gram] += 1

"""
How we could split object classes based on occurrence in questions and answers?
1. Generalization to rare words: have rare objects in the test set.
This way of spliting classes well matches with the motivation:
    Collecting VQA training data for all rare object classes are difficult, but
    there are rich set of object annotations.
    So transfer to rare words make sense.
Why should we keep very frequent words:
    This is to run reasoning well. VQA training data with very rare words only
    teaches the biases. But rich words appears with various forms of questions
    and this help to learn real reasoning, not a bias.
* How we will split data:
    1. occurence >= threshold_1: always training set
    2. threshold_1 > occurrence >= threshold_2: random split
    3. threshold_2 > occurrence: always testing set

c.f)
Among 4810 classes
top 500 occurrence object classes have more than 500 occurrences
if we pick top 1650-th occurrece object, its occurrene is around 50.
Therefore, with threshold_1 == 500, threshold_2 == 50,
around 487 classes are always 'train'
around 1167 classes are randomly splited into 'train' and 'test'
around 3156 classes are randomly splited into 'train-reserve' and 'test'
(train-reserve could be used for training or not, based on the experiment result)
"""
obj_grp1 = [name for name, occ in occurrence.items()
            if occ >= config.occ_thres_1]
obj_grp2 = [name for name, occ in occurrence.items()
            if config.occ_thres_1 > occ >= config.occ_thres_2]
obj_grp3 = [name for name, occ in occurrence.items()
            if config.occ_thres_2 > occ]
RANDOM_STATE.shuffle(obj_grp1)
RANDOM_STATE.shuffle(obj_grp2)
half2 = len(obj_grp2) / 2
RANDOM_STATE.shuffle(obj_grp3)
half3 = len(obj_grp3) / 2
objects_split = {
    'train': obj_grp1 + obj_grp2[:half2],
    'train-reserve': obj_grp3[:half3],
    'test': obj_grp2[half2:] + obj_grp3[half3:],
}

qids = qid2anno.keys()
print('Mark test QA')
print('Mark train-reserve QA')
print('Split test-val / test')
print('Split train / val')
