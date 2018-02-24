import json
import os

from tqdm import tqdm

import tools

ANNO_DIR = 'VisualGenome/annotations'
ANNO_FILE = {
    'objects': 'objects.json',
    'attributes': 'attributes.json',
    'relationships': 'relationships.json',
    'region_descriptions': 'region_descriptions.json',
}
VOCAB_PATH = 'preprocessed/vocab.json'

anno = {}
for key, fn in tqdm(ANNO_FILE.items(), desc='reading annotations'):
    anno[key] = json.load(open(os.path.join(ANNO_DIR, fn), 'r'))
vocab = json.load(open(VOCAB_PATH, 'r'))
vocab_set = set(vocab['vocab'])


def check_and_add(name, name_list):
    name = tools.clean_description(name)
    if len(name) > 0 and all([n in vocab_set for n in name.split()]):
        name_list.append(name)

objects = []
for entry in tqdm(anno['objects'], desc='objects'):
    for obj in entry['objects']:
        if 'name' in obj: check_and_add(obj['name'], objects)
        if 'names' in obj:
            for name in obj['names']:
                check_and_add(name, objects)

attributes = []
for entry in tqdm(anno['attributes'], desc='attributes'):
    for attr in entry['attributes']:
        if 'attributes' in attr:
            for attr_name in attr['attributes']:
                check_and_add(attr_name, attributes)

relationships = []
for entry in tqdm(anno['relationships'], desc='relationships'):
    for rel in entry['relationships']:
        if 'predicate' in rel:
            check_and_add(rel['predicate'], relationships)

descriptions = []
for entry in tqdm(anno['region_descriptions'], desc='descriptions'):
    for region in entry['regions']:
        check_and_add(region['phrase'], descriptions)
