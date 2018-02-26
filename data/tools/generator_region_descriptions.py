"""
Generator for region_descriptions

Generate region_description data for training vlmap.
"""
import argparse
import h5py
import json
import os
import numpy as np

from tqdm import tqdm

import tools

ANNO_DIR = 'VisualGenome/annotations'
ANNO_FILE = {
    'region_descriptions': 'region_descriptions.json',
}
VOCAB_PATH = 'preprocessed/vocab.json'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_name', type=str, default='region_descriptions')
args = parser.parse_args()

args.dir_name = os.path.join('preprocessed', args.dir_name)

if not os.path.exists(args.dir_name):
    os.makedirs(args.dir_name)
else:
    raise ValueError('The directory {} already exists. Do not overwrite'.format(
        args.dir_name))

args.hdf5_file = os.path.join(args.dir_name, 'data.hdf5')
args.ids_file = os.path.join(args.dir_name, 'id.txt')
args.stats_file = os.path.join(args.dir_name, 'stats.txt')
args.descriptions_file = os.path.join(args.dir_name, 'descriptions.txt')

print('Reading annotations..')
anno = {}
anno['region_descriptions'] = \
    json.load(open(os.path.join(ANNO_DIR,
                                ANNO_FILE['region_descriptions']), 'r'))
print('Done.')

vocab = json.load(open(VOCAB_PATH, 'r'))
vocab_set = set(vocab['vocab'])


def clean_phrase(phrase):
    phrase = tools.clean_description(phrase)
    if len(phrase) > 0 and all([n in vocab_set for n in phrase.split()]):
        return phrase
    else: return ''


def phrase2intseq(phrase):
    return np.array([vocab['dict'][n] for n in phrase.split()], dtype=np.int32)

f = h5py.File(args.hdf5_file, 'w')
id_file = open(args.ids_file, 'w')

num_train_image = 80000
num_test_image = 18077
num_val_image = 10000

num_train = 0
num_test = 0
num_val = 0

cnt = 0
max_length = 0
descriptions = []
for image_cnt, entry in enumerate(tqdm(anno['region_descriptions'],
                                       desc='region_descriptions')):
    for region in entry['regions']:
        phrase = region['phrase']

        phrase = clean_phrase(phrase)
        if phrase == '': continue
        descriptions.append(phrase)

        phrase = np.array(phrase2intseq(phrase), dtype=np.int32)

        max_length = max(max_length, len(phrase))

        image_id = region['image_id']
        id = 'descriptions{:08d}_imageid{}_length{}'.format(
            cnt, image_id, len(phrase))

        if str(image_id) in f: image_grp = f[str(image_id)]
        else: image_grp = f.create_group(str(image_id))

        grp = image_grp.create_group(id)
        grp['image_id'] = image_id
        grp['description'] = phrase
        grp['region_id'] = region['region_id']
        grp['x'], grp['y'] = region['x'], region['y']
        grp['w'], grp['h'] = region['width'], region['height']

        id_file.write(str(image_id) + ' ' + id + '\n')
        cnt += 1
        if image_cnt < num_train_image:
            num_train = cnt
        elif image_cnt < num_train_image + num_test_image:
            num_test = cnt - num_train
        else:
            num_val = cnt - num_train - num_test

set_descriptions = list(set(descriptions))

grp = f.create_group('data_info')
grp['max_length'] = max_length
grp['num_data'] = cnt
grp['num_train'] = num_train
grp['num_test'] = num_test
grp['num_val'] = num_val
grp['num_images'] = len(anno['region_descriptions'])
grp['num_train_image'] = num_train_image
grp['num_test_image'] = num_test_image
grp['num_val_image'] = num_val_image
grp['num_unique_descriptions'] = len(set_descriptions)

id_file.close()
f.close()

stat_file = open(args.stats_file, 'w')
stat_file.write('num_data: {}\n'.format(cnt))
stat_file.write('num_train: {}\n'.format(num_train))
stat_file.write('num_test: {}\n'.format(num_test))
stat_file.write('num_val: {}\n'.format(num_val))
stat_file.write('num_images: {}\n'.format(len(anno['region_descriptions'])))
stat_file.write('num_train_image: {}\n'.format(num_train_image))
stat_file.write('num_test_image: {}\n'.format(num_test_image))
stat_file.write('num_val_image: {}\n'.format(num_val_image))
stat_file.write('num_unique_descriptions: {}\n'.format(len(set_descriptions)))
stat_file.write('max_length: {}\n'.format(max_length))
stat_file.close()

descriptions_file = open(args.descriptions_file, 'w')
for name in set_descriptions:
    descriptions_file.write(name + '\n')
descriptions_file.close()

print('description dataset is created: {}'.format(args.dir_name))
