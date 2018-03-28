import argparse
import h5py
import json
import os
import re
import numpy as np

from tqdm import tqdm
from util import log


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
parser.add_argument('--save_split_dir', type=str,
                    default='data/preprocessed/vqa_v2/new_qa_split')
config = parser.parse_args()

vocab = json.load(open(config.vocab_path, 'r'))

config.save_split_dir += '_thres1_{}'.format(config.occ_thres_1)
config.save_split_dir += '_thres2_{}'.format(config.occ_thres_2)

if not os.path.exists(config.save_split_dir):
    log.warn('Create directory: {}'.format(config.save_split_dir))
    os.makedirs(config.save_split_dir)
else:
    raise ValueError(
        'The directory {} already exists. Do not overwrite.'.format(
            config.save_split_dir))


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

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def split_with_punctuation(string):
    return re.findall(r"'s+|[\w]+|[.,!?;-]", string)

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

for qid in tqdm(qid2anno.keys(), desc='tokenize QA'):
    anno = qid2anno[qid]
    qid2anno[qid]['q_tokens'] = split_with_punctuation(
        anno['question'].lower())
    qid2anno[qid]['a_tokens'] = split_with_punctuation(
        preprocess_answer(anno['multiple_choice_answer'].lower()))
    processed_answers = []
    for answer in anno['answers']:
        a_tokens = split_with_punctuation(
            preprocess_answer(answer['answer'].lower()))
        processed_answers.append(' '.join(a_tokens))
    qid2anno[qid]['processed_answers'] = processed_answers

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
qid2ngrams = {}
for qid, anno in tqdm(qid2anno.items(), desc='count object occurrence'):
    q_tokens = anno['q_tokens']
    a_tokens = anno['a_tokens']
    ngrams = []
    for n in range(1, min(len(q_tokens), max_name_len)):
        ngrams.extend(get_ngrams(q_tokens, n))
    for n in range(1, min(len(a_tokens), max_name_len)):
        ngrams.extend(get_ngrams(a_tokens, n))
    ngrams = list(set(ngrams))
    for ngram in ngrams:
        if ngram in occurrence: occurrence[ngram] += 1
    qid2ngrams[qid] = ngrams

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
log.warn('Split objects')
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
objects_split_set = {key: set(val) for key, val in objects_split.items()}
log.infov('train object: {}'.format(len(objects_split['train'])))
log.info('ex)')
for obj in objects_split['train'][:5]: log.info(obj)

log.infov('train-reserve object: {}'.format(len(objects_split['train-reserve'])))
log.info('ex)')
for obj in objects_split['train-reserve'][:5]: log.info(obj)

log.infov('test object: {}'.format(len(objects_split['test'])))
log.info('ex)')
for obj in objects_split['test'][:5]: log.info(obj)


def filter_qids_by_object_split(qids, split):
    filtered_qids = []
    for qid in tqdm(qids, desc='mark {} QA'.format(split)):
        ngrams = qid2ngrams[qid]
        is_target_split = False
        for ngram in ngrams:
            if ngram in objects_split_set[split]:
                is_target_split = True
                break
        if is_target_split: filtered_qids.append(qid)
    return filtered_qids

qids = qid2anno.keys()

log.warn('Mark test QA')
test_qids = filter_qids_by_object_split(qids, 'test')
left_qids = list(set(qids) - set(test_qids))
log.infov('{} question ids are marked for test objects'.format(len(test_qids)))

log.warn('Mark train-reserve QA')
train_reserve_qids = filter_qids_by_object_split(left_qids, 'train-reserve')
train_qids = list(set(left_qids) - set(train_reserve_qids))
log.infov('{} question ids are marked for train-reserve objects'.format(
    len(train_reserve_qids)))
log.infov('{} question ids are marked for train objects'.format(
    len(train_qids)))

log.warn('Shuffle qids')
RANDOM_STATE.shuffle(train_qids)
RANDOM_STATE.shuffle(train_reserve_qids)
RANDOM_STATE.shuffle(test_qids)

train_90p = len(train_qids) * 90 / 100
train_reserve_90p = len(train_reserve_qids) * 90 / 100
test_80p = len(test_qids) * 80 / 100

qid_splits = {}
log.warn('Split test-val / test')
qid_splits['test-val'] = test_qids[test_80p:]
qid_splits['test'] = test_qids[:test_80p]
log.infov('test: {}, test-val: {}'.format(
    len(qid_splits['test']), len(qid_splits['test-val'])))

log.warn('Split train / val')
qid_splits['val'] = train_qids[train_90p:]
qid_splits['train'] = train_qids[:train_90p]
log.infov('train: {}, val: {}'.format(
    len(qid_splits['train']), len(qid_splits['val'])))
qid_splits['val-reserve'] = train_reserve_qids[train_reserve_90p:]
qid_splits['train-reserve'] = train_reserve_qids[:train_reserve_90p]
log.infov('train-reserve: {}, val-reserve: {}'.format(
    len(qid_splits['train-reserve']), len(qid_splits['val-reserve'])))

# used image ids
used_image_paths = []
for anno in tqdm(qid2anno.values(), desc='construct used_image_ids'):
    used_image_paths.append(anno['image_path'])
used_image_paths = list(set(used_image_paths))
log.infov('used_image_paths: {}'.format(len(used_image_paths)))

"""
What to save:
    - used image ids (for efficient feature extraction)
    - object splits (train , train-reserve, test)
    - qa splits qids (train, val, train-reserve, val-reserve, test-val, test)
    - annotations: merged annotations is saved for evaluation / future usage
"""
with open(os.path.join(config.save_split_dir, 'used_image_path.txt'), 'w') as f:
    for image_path in used_image_paths:
        f.write(image_path + '\n')
json.dump(objects_split, open(os.path.join(
    config.save_split_dir, 'object_split.json'), 'w'))
json.dump(qid_splits, open(os.path.join(
    config.save_split_dir, 'qa_split.json'), 'w'))
json.dump(qid2anno, open(os.path.join(
    config.save_split_dir, 'merged_annotations.json'), 'w'))
log.warn('output saving is done.')
