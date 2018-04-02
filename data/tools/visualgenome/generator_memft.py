"""
Generator for on memory pre-training of vlmap

each entry contains bounding boxes and object, attribute, description annotations.
36 bounding boxes are extracted using "bottom_up attention".

pre-training aim to two tasks: attention / answer:
    - attention: language -> box relevant score (36 scalar)
    - answer: 36 boxes -> pooled vector -> answer

For each data, following information should be encoded:
    - non-zero scores for each bounding box
    - weight for each box for feature pooling
    - object, attribute, description annotations
"""
import argparse
import h5py
import json
import os
import numpy as np

from collections import Counter
from tqdm import tqdm

from data.tools import tools

ANNO_FILES = {
    'object': 'data/VisualGenome/annotations/objects.json',
    'attribute': 'data/VisualGenome/annotations/attributes.json',
    'caption': 'data/VisualGenome/annotations/region_descriptions.json',
}
IMAGE_DATA_PATH = 'data/VisualGenome/annotations/image_data.json'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vocab_path', type=str,
                    default='data/preprocessed/new_vocab50.json', help=' ')
parser.add_argument('--bottomup_data_dir', type=str,
                    default='data/VisualGenome/bottomup_feature_36', help=' ')
parser.add_argument('--dir_name', type=str, default='memft_all', help=' ')
parser.add_argument('--min_occurrence', type=int, default=20, help=' ')
parser.add_argument('--max_description_length', type=int, default=10, help=' ')
config = parser.parse_args()

config.dir_name = os.path.join('data/preprocessed/visualgenome', config.dir_name)
config.dir_name += '_{}'.format(
    config.vocab_path.replace('data/preprocessed/', '').replace('.json', ''))
config.dir_name += '_{}_min_occ{}'.format(
    config.vocab_path.replace('data/preprocessed/', '').replace('.json', ''),
    config.min_occurrence)
if config.max_description_length > 0:
    config.dir_name += '_max_len{}'.format(config.max_description_length)

#if not os.path.exists(args.dir_name): os.makedirs(args.dir_name)
#else: raise ValueError('Do not overwrite {}'.format(args.dir_name))

image_data = json.load(open(IMAGE_DATA_PATH, 'r'))

vocab = json.load(open(config.vocab_path, 'r'))
vocab_set = set(vocab['vocab'])


def check_name(name):
    name = tools.clean_answer_word(name)
    passed = len(name) > 0 and all([n in vocab_set for n in name.split()])
    return passed, name


def check_and_add(name, name_list):
    name = tools.clean_answer_word(name)
    if len(name) > 0 and all([n in vocab_set for n in name.split()]):
        name_list.append(name)


def check_caption(caption):
    caption = tools.clean_description(caption)
    passed = len(caption) > 0 and all([n in vocab_set for n in caption.split()])
    return passed, caption


def check_and_add_description(name, name_list):
    name = tools.clean_description(name)
    if len(name) > 0 and all([n in vocab_set for n in name.split()]):
        name_list.append(name)

annotations = {}
for key, anno_fn in tqdm(ANNO_FILES.items(), desc='loading anno..'):
    annotations[key] = json.load(open(anno_fn, 'r'))

config.vfeat_path = os.path.join(config.bottomup_data_dir,
                                 'vfeat_bottomup_36.hdf5')
config.image_info_path = os.path.join(config.bottomup_data_dir,
                                      'image_info.json')

vfeat_h5 = h5py.File(config.vfeat_path, 'r')

image_info = json.load(open(config.image_info_path, 'r'))
image_id2idx = image_info['image_id2idx']

"""
process objects
"""
obj_blacklist = set(['is', 'it', 'red', 'yellow', 'black', 'blue', 'green', 'pink',
                     'orange', 'purple', 'brown', 'white', 'gray', 'grey',
                     'gold', 'silver', 'tall', 'long', 'short', 'big', 'small',
                     'left', 'right', 'up', 'down', 'middle'])

color_words = set(['red', 'yellow', 'black', 'blue', 'green', 'pink',
                   'orange', 'purple', 'brown', 'white', 'gray', 'grey',
                   'gold', 'silver'])


def strip_color(name):
    tokens = name.split()
    if len(tokens) == 2:
        if tokens[0] in color_words:
            return tokens[0], tokens[1]
        else:
            return None, name
    else:
        return None, name


def strip_number(name):
    tokens = name.split()
    if len(tokens) == 2:
        if str.isdigit(str(tokens[0])):
            return tokens[0], tokens[1]
        else:
            return None, name
    else:
        return None, name

freq_obj = []
for entry in tqdm(annotations['object'], desc='process obj1'):
    for e in entry['objects']:
        is_passed, name = check_name(e['names'][0])
        if is_passed and (name not in obj_blacklist) and (not str.isdigit(str(name))):
            color_w, name = strip_color(name)
            digit_w, name = strip_number(name)
            e['processed_name'] = name
            freq_obj.append(name)
freq_obj = Counter(freq_obj)
freq_obj = dict(freq_obj.most_common()[:3000])  # use top 3000 objects
freq_obj_set = set(freq_obj.keys())

for entry in tqdm(annotations['object'], desc='process obj2'):
    for e in entry['objects']:
        if 'processed_name' in e:
            if e['processed_name'] not in freq_obj_set:
                del e['processed_name']

image_id2objects = {}
for entry in tqdm(annotations['object'], desc='process obj3'):
    image_id = entry['image_id']
    if image_id not in image_id2objects:
        image_id2objects[image_id] = []
    for e in entry['objects']:
        if 'processed_name' in e:
            image_id2objects[image_id].append(e)

"""
process attributes
"""
obj2attr_list = {}
for entry in tqdm(annotations['attribute'], desc='process attr1'):
    for e in entry['attributes']:
        if 'names' not in e or len(e['names']) != 1:
            continue
        if 'attributes' not in e:
            continue
        passed, processed_name = check_name(e['names'][0])
        if (not passed) or (processed_name not in freq_obj_set):
            continue
        processed_attributes = set()
        color_w, processed_name = strip_color(processed_name)
        if color_w is not None:
            processed_attributes.add(color_w)
        digit_w, processed_name = strip_number(processed_name)
        if digit_w is not None:
            processed_attributes.add(digit_w)
        for attr in e['attributes']:
            passed, processed_attr = check_name(attr)
            if passed and (processed_attr not in freq_obj_set):
                processed_attributes.add(processed_attr)
        processed_attributes = list(processed_attributes)
        if len(processed_attributes) == 0:
            continue
        e['processed_name'] = processed_name
        e['processed_attributes'] = processed_attributes
        if processed_name not in obj2attr_list:
            obj2attr_list[processed_name] = set()
        for attr in processed_attributes:
            obj2attr_list[processed_name].add(attr)

freq_attr = []
attr_blacklist = set(['is'])
for entry in tqdm(annotations['attribute'], desc='process attr2'):
    for e in entry['attributes']:
        if 'processed_name' not in e: continue
        if 'processed_attributes' not in e: continue
        for attr in e['processed_attributes']:
            if attr not in attr_blacklist:
                freq_attr.append(attr)
freq_attr = Counter(freq_attr)
freq_attr = dict(freq_attr.most_common()[:1000])  # use top 1000 attributes
freq_attr_set = set(freq_attr.keys())

obj2attr_list = {}
for entry in tqdm(annotations['attribute'], desc='process attr3'):
    for e in entry['attributes']:
        if 'processed_name' not in e: continue
        if 'processed_attributes' not in e: continue
        name = e['processed_name']
        if name not in obj2attr_list:
            obj2attr_list[name] = set()
        processed_attributes = set()
        for attr in e['processed_attributes']:
            if attr in freq_attr_set:
                obj2attr_list[name].add(attr)
                processed_attributes.add(attr)
        processed_attributes = list(processed_attributes)
        if len(processed_attributes) == 0:
            del e['processed_attributes']
        else: e['processed_attributes'] = processed_attributes

image_id2attrs = {}
for entry in tqdm(annotations['attribute'], desc='process attr4'):
    image_id = entry['image_id']
    if image_id not in image_id2attrs:
        image_id2attrs[image_id] = []
    for e in entry['attributes']:
        if 'processed_name' in e and 'processed_attributes' in e:
            image_id2attrs[image_id].append(e)

"""
process descriptions
"""
obj_attr_set = freq_obj_set | freq_attr_set

vocab2obj = {}
for obj in freq_obj_set:
    for t in obj.split():
        if t not in vocab2obj:
            vocab2obj[t] = set()
        vocab2obj[t].add(obj)

for entry in tqdm(annotations['caption'], desc='process caption1'):
    for e in entry['regions']:
        if 'phrase' not in e:
            continue
        passed, caption = check_caption(e['phrase'])
        if not passed:
            continue
        e['caption'] = caption
        obj_candidates = set()
        for t in caption.split():
            if t in vocab2obj:
                obj_candidates = obj_candidates | vocab2obj[t]

        matched_obj_candidates = set()
        for obj in obj_candidates:
            if caption.count(obj) > 0:
                matched_obj_candidates.add(obj)

        cand_list = list(matched_obj_candidates)
        longest = [True] * len(cand_list)
        for i in range(len(cand_list)):
            for j in range(len(cand_list)):
                if i != j and cand_list[j].count(cand_list[i]) > 0:
                    longest[i] = False

