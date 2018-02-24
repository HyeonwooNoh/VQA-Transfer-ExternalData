import json
import os

from tqdm import tqdm

import tools

ANNO_DIR = 'VisualGenome/annotations'
ANNO_FILE = {
    'objects': 'objects.json',
    'attributes': 'attributes.json',
    'region_descriptions': 'region_descriptions',
}
VOCAB_PATH = 'preprocessed/vocab.json'

anno = {}
for key, fn in tqdm(ANNO_FILE.items(), desc='reading annotations'):
    anno[key] = json.load(open(os.path.join(ANNO_DIR, fn), 'r'))
vocab = json.load(open(VOCAB_PATH, 'r'))


def check_and_add(name, name_list):
    name = tools.clean_description(name)
    if all([n in vocab['vocab'] for n in name.split()]):
        name_list.append(name)

objects = []
for entry in tqdm(anno['objects'], desc='objects'):
    for obj in entry['objects']:
        if 'name' in obj: check_and_add(obj['name'], objects)
        if 'names' in obj:
            for name in obj['names']:
                check_and_add(name, objects)

attributes = []
attr_objs = []
for entry in tqdm(anno['attributes'], desc='attributes'):
    for attr in entry['attributes']:
        if 'attributes' in attr:
            for attr_name in attr['attributes']:
                check_and_add(attr_name, attributes)
        if 'attributes' in attr and 'name' in attr:
            name = ' '.join(attr['attributes']) + ' ' + attr['name']
            check_and_add(name, attr_objs)
        if 'attributes' in attr and 'names' in attr:
            for name in attr['names']:
                name = ' '.join(attr['attributes']) + ' ' + name
                check_and_add(name, attr_objs)

descriptions = []
for entry in tqdm(anno['region_descriptions'], desc='descriptions'):
    for region in entry['regions']:
        check_and_add(region['phrase'], descriptions)
