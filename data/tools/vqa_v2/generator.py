import argparse
import h5py
import json
import os
import re
import numpy as np

from tqdm import tqdm
from util import log

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--qa_split_dir', type=str,
                    default='data/preprocessed/vqa_v2'
                    '/qa_split_thres1_500_thres2_50', help=' ')
parser.add_argument('--use_train_reserve', action='store_true', default=False)
config = parser.parse_args()

config.data_dir = os.path.join(config.qa_split_dir, 'data')
config.data_path = os.path.join(config.data_dir, 'data.hdf5')
config.id_path = os.path.join(config.data_dir, 'id.txt')
if not os.path.exists(config.data_dir):
    log.warn('create directory: {}'.format(config.data_dir))
    os.makedirs(config.data_dir)
else:
    raise ValueError(
        'The directory {} already exists. Do not overwrite.'.format(
            config.data_dir))

log.info('loading vocab')
vocab = json.load(open(
    os.path.join(config.qa_split_dir, 'vocab.json'), 'r'))
log.info('loading frequent answers')
freq_ans = json.load(open(
    os.path.join(config.qa_split_dir, 'frequent_answers.json'), 'r'))
freq_ans_set = set(freq_ans)
log.info('loading merged_annotations')
qid2anno = json.load(open(os.path.join(
    config.qa_split_dir, 'merged_annotations.json'), 'r'))
log.info('loading qa_split')
qa_split = json.load(open(os.path.join(
    config.qa_split_dir, 'qa_split.json'), 'r'))
log.info('loading is done')

"""
How to store answers:
    - in ['data_info'], intseq and intseq_len of all answer candidates are saved
    - for each data point, index of answer candidate is saved as a ground truth
How to store questions:
    - max_q_len is saved in ['data_info']
    - q_intseq is saved for each question. (q_intseq_len is not saved because
      they can be easily obtained from q_intseq)
Data statistics:
    - num_train, num_val, num_test_val, num_test
"""
f = h5py.File(config.data_path, 'w')

if config.use_train_reserve:
    qa_split['train'].extend(qa_split['train-reserve'])
    qa_split['val'].extend(qa_split['val-reserve'])

# encode frequent answers
intseq_ans = []
intseq_ans_len = []
for ans in freq_ans:
    intseq_ans.append([vocab['dict'][t] for t in ans.split()])
    intseq_ans_len.append(len(intseq_ans[-1]))

np_intseq_ans_len = np.array(intseq_ans_len, dtype=np.int32)
max_ans_len = np_intseq_ans_len.max()

np_intseq_ans = np.zeros([len(intseq_ans), max_ans_len], dtype=np.int32)
for i, intseq in enumerate(intseq_ans):
    np_intseq_ans[i, :len(intseq)] = np.array(intseq, dtype=np.int32)
ans_dict = {ans: i for i, ans in enumerate(freq_ans)}

# add frequent answer intseqs to data_info
data_info = f.create_group('data_info')
data_info['intseq_ans'] = np_intseq_ans
data_info['intseq_ans_len'] = np_intseq_ans_len

"""
For training set QA:
    - if answer is not in freq_ans_set, ignore the example
    - if the vocab is not in GloVe, mark it <unk>
For testing set QA:
    - if answer is not in freq_ans_set, mark it N+1-th answer
    - if the vocab is not in GloVe, mark it <unk>
"""
used_qid = {
    'train': [],
    'val': [],
    'test-val': [],
    'test': [],
}
for split in ['train', 'val']:
    for qid in tqdm(qa_split[split], desc='process {} qids'.format(split)):
        anno = qid2anno[str(qid)]
        ans = ' '.join(anno['a_tokens'])
        if ans not in freq_ans_set:
            continue  # for train set, we ignore qa with rare answer
        ans_id = ans_dict[ans]

        unk_id = vocab['dict']['<unk>']
        q_intseq = [vocab['dict'].get(t, unk_id) for t in anno['q_tokens']]
        q_intseq = np.array(q_intseq, dtype=np.int32)
        used_qid[split].append(qid)

        grp = f.create_group(str(qid))
        grp['answer_id'] = ans_id
        grp['question_intseq'] = q_intseq
        grp['image_path'] = anno['image_path']

for split in ['test-val', 'test']:
    for qid in tqdm(qa_split[split], desc='process {} qids'.format(split)):
        anno = qid2anno[str(qid)]
        ans = ' '.join(anno['a_tokens'])
        ans_id = ans_dict.get(ans, len(freq_ans))

        unk_id = vocab['dict']['<unk>']
        q_intseq = [vocab['dict'].get(t, unk_id) for t in anno['q_tokens']]
        q_intseq = np.array(q_intseq, dtype=np.int32)
        used_qid[split].append(qid)

        grp = f.create_group(str(qid))
        grp['answer_id'] = ans_id
        grp['question_intseq'] = q_intseq
        grp['image_path'] = anno['image_path']

log.warn('write to data_info')
data_info['num_train'] = len(used_qid['train'])
data_info['num_val'] = len(used_qid['val'])
data_info['num_test-val'] = len(used_qid['test-val'])
data_info['num_test'] = len(used_qid['test'])
f.close()

log.warn('write to id file')
fid = open(config.id_path, 'w')
for key in ['train', 'val', 'test-val', 'test']:
    for qid in used_qid:
        fid.write(str(qid) + '\n')
fid.close()
log.infov('data file is saved to: {}'.format(config.data_path))
log.infov('id file is saved to: {}'.format(config.id_path))
