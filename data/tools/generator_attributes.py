"""
Generator for attibutes

Generate attribute data for training vlmap.
"""
import argparse
import h5py
import json
import os
import numpy as np

from collections import Counter
from tqdm import tqdm

import tools

ANNO_DIR = 'VisualGenome/annotations'
ANNO_FILE = {
    'attributes': 'attributes.json',
}
VOCAB_PATH = 'preprocessed/vocab.json'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_name', type=str, default='attributes')
parser.add_argument('--min_occurrence', type=int, default=20)
args = parser.parse_args()

args.dir_name = os.path.join('preprocessed', args.dir_name)
args.dir_name += '_min_occ{}'.format(args.min_occurrence)

if not os.path.exists(args.dir_name):
    os.makedirs(args.dir_name)
else:
    raise ValueError('The directory {} already exists. Do not overwrite'.format(
        args.dir_name))

args.hdf5_file = os.path.join(args.dir_name, 'data.hdf5')
args.ids_file = os.path.join(args.dir_name, 'id.txt')
args.stats_file = os.path.join(args.dir_name, 'stats.txt')
args.attributes_file = os.path.join(args.dir_name, 'attributes.txt')

print('Reading annotations..')
anno = {}
anno['attributes'] = json.load(open(os.path.join(ANNO_DIR, ANNO_FILE['attributes']), 'r'))
print('Done.')

vocab = json.load(open(VOCAB_PATH, 'r'))
vocab_set = set(vocab['vocab'])


def clean_name(name):
    name = tools.clean_description(name)
    if len(name) > 0 and all([n in vocab_set for n in name.split()]):
        return name
    else: return ''


def check_and_add(name, name_list):
    name = tools.clean_description(name)
    if len(name) > 0 and all([n in vocab_set for n in name.split()]):
        name_list.append(name)


def name2intseq(name):
    return np.array([vocab['dict'][n] for n in name.split()], dtype=np.int32)

attributes = []
for entry in tqdm(anno['attributes'], desc='attributes'):
    for attr in entry['attributes']:
        if 'name' in attr: check_and_add(attr['name'], attributes)
        if 'names' in attr:
            for name in attr['names']:
                check_and_add(name, attributes)

attribute_count = Counter(attributes)
thr_attributes_set = set([o for o in list(set(attributes))
                          if attribute_count[o] >= args.min_occurrence])

f = h5py.File(args.hdf5_file, 'w')
id_file = open(args.ids_file, 'w')

cnt = 0
max_name_length = 0
max_num_names = 0
for entry in tqdm(anno['attributes'], desc='attributes'):
    image_id = entry['image_id']
    image_grp = f.create_group(str(image_id))
    for attr in entry['attributes']:
        names = []
        if 'name' in attr:
            names.append(attr['name'])
        if 'names' in attr:
            names.extend(attr['names'])

        name_len = []
        names_intseq = []
        for name in names:
            name = clean_name(name)
            if name == '' or name not in thr_attributes_set:
                continue
            intseq = name2intseq(name)
            name_len.append(len(intseq))
            names_intseq.append(intseq)

        if len(names_intseq) == 0:
            continue

        names = np.zeros([len(names_intseq), max(name_len)], dtype=np.int32)
        for i, intseq in enumerate(names_intseq):
            names[i][:len(intseq)] = intseq
        name_len = np.array(name_len, dtype=np.int32)

        max_num_names = max(max_num_names, names.shape[0])
        max_name_length = max(max_name_length, names.shape[1])

        id = 'attributes{:08d}_imageid{}_numname{}_maxnamelen{}'.format(
            cnt, image_id, names.shape[0], names.shape[1])

        grp = image_grp.create_group(id)
        grp['image_id'] = image_id
        grp['names'] = names
        grp['name_len'] = name_len
        grp['h'], grp['w'] = attr['h'], attr['w']
        grp['y'], grp['x'] = attr['y'], attr['x']
        grp['object_id'] = attr['object_id']

        id_file.write(str(image_id) + ' ' + id + '\n')
        cnt += 1

thr_attribute_set_intseq = np.zeros([len(thr_attributes_set), max_name_length],
                                    dtype=np.int32)
thr_attribute_set_intseq_len = np.zeros([len(thr_attributes_set)], dtype=np.int32)
for i, name in enumerate(list(thr_attributes_set)):
    intseq = name2intseq(name)
    thr_attribute_set_intseq[i, :len(intseq)] = intseq
    thr_attribute_set_intseq_len[i] = len(intseq)

grp = f.create_group('data_info')
grp['max_name_length'] = max_name_length
grp['max_num_names'] = max_num_names
grp['num_data'] = cnt
grp['num_unique_attributes'] = len(thr_attributes_set)
grp['attributes_intseq'] = thr_attribute_set_intseq
grp['attributes_intseq_len'] = thr_attribute_set_intseq_len
grp['min_occurrence'] = args.min_occurrence

id_file.close()
f.close()

stat_file = open(args.stats_file, 'w')
stat_file.write('num_data: {}\n'.format(cnt))
stat_file.write('num_unique_attributes: {}\n'.format(len(thr_attributes_set)))
stat_file.write('max_num_names: {}\n'.format(max_num_names))
stat_file.write('max_name_length: {}\n'.format(max_name_length))
stat_file.write('min_occurrence: {}\n'.format(args.min_occurrence))
stat_file.close()

attributes_file = open(args.attributes_file, 'w')
for name in list(thr_attributes_set):
    attributes_file.write(name + '\n')
attributes_file.close()

print('Attribute dataset is created: {}'.format(args.dir_name))
