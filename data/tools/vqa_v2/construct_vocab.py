import argparse
import collections
import json
import os

from tqdm import tqdm

from util import log

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--glove_vocab_path', type=str,
                    default='data/preprocessed/glove_vocab.json', help=' ')
parser.add_argument('--qa_split_dir', type=str,
                    default='data/preprocessed/vqa_v2'
                    '/qa_split_thres1_500_thres2_50', help=' ')
parser.add_argument('--answer_set_limit', type=int, default=3000, help=' ')
config = parser.parse_args()

log.info('loading merged_annotations..')
qid2anno = json.load(open(os.path.join(
    config.qa_split_dir, 'merged_annotations.json'), 'r'))
log.info('loading glove_vocab..')
glove_vocab = json.load(open(config.glove_vocab_path, 'r'))
log.info('done')

"""
Filtering qa:
    - count answer occurrences and filter rare answers
      (filtering answer candidates is for efficiency of answer classification)
    - make answer set with only frequent answers
    - make sure that every chosen answers consist of glove vocabs
    - all questions are used for vocab construction
*: When making GT, let's make N(answer) + 1 slots and if answer is not in answer
set, mark gt to N(answer) + 1 slot (only for testing data). For training data,
we will just ignore qa with rare answers.
"""

glove_vocab_set = set(glove_vocab['vocab'])

answers = list()
for anno in tqdm(qid2anno.values(), desc='count answers'):
    answers.append(' '.join(anno['a_tokens']))
answer_counts = collections.Counter(answers)

ans_in_order = list(zip(*answer_counts.most_common())[0])
ans_in_order_glove = []
for ans in ans_in_order:
    in_glove = True
    for t in ans.split():
        if t not in glove_vocab_set:
            in_glove = False
            break
    if in_glove:
        ans_in_order_glove.append(ans)

freq_ans = ans_in_order_glove[:config.answer_set_limit]
freq_ans_set = set(freq_ans)

"""
For training set QA:
    - if answer is not in freq_ans_set, ignore the example
    - if the vocab is not in GloVe, mark it <unk>
For testing set QA:
    - if answer is not in freq_ans_set, mark it N+1-th answer
    - if the vocab is not in GloVe, mark it <unk>
"""
q_vocab = set()
for anno in tqdm(qid2anno.values(), desc='count q_vocab'):
    for t in anno['q_tokens']: q_vocab.add(t)
q_vocab = q_vocab & glove_vocab_set

a_vocab = set()
for ans in tqdm(freq_ans, desc='count a_vocab'):
    for t in ans.split(): a_vocab.add(t)
if len(a_vocab - glove_vocab_set) > 0:
    raise RuntimeError('Somethings wrong: a_vocab is already filtered')
qa_vocab = q_vocab | a_vocab
qa_vocab = list(qa_vocab)
qa_vocab.append('<s>')
qa_vocab.append('<e>')
qa_vocab.append('<unk>')

"""
How to store vocab:
    - ['vocab'] = ['yes', 'no', 'apple', ...]
    - ['dict'] = {'yes': 0, 'no: 1, 'apple': 2, ...}
    - in a json format
"""
save_vocab = {}
save_vocab['vocab'] = qa_vocab
save_vocab['dict'] = {v: i for i, v in enumerate(qa_vocab)}

vocab_path = os.path.join(config.qa_split_dir, 'vocab.json')
log.warn('save vocab: {}'.format(vocab_path))
json.dump(save_vocab, open(vocab_path, 'w'))

freq_ans_path = os.path.join(config.qa_split_dir, 'frequent_answers.json')
log.warn('save frequent answers: {}'.format(freq_ans_path))
json.dump(freq_ans, open(freq_ans_path, 'w'))

log.warn('done')
