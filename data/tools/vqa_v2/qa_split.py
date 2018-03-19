import argparse
import h5py
import json
import os

from tqdm import tqdm


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

