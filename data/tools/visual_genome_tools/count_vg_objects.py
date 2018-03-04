"""Count objects in VisualGenome datasets:
"""
import json
import os

from tqdm import tqdm

ANNO_DIR = 'VisualGenome/annotations'
ANNO_FILE = {
    'objects': 'objects.json',
    'attributes': 'attributes.json',
    'relationships': 'relationships.json',
}
OUTPUT_PATH = 'preprocessed/object_count.json'

if os.path.exists(OUTPUT_PATH):
    raise RuntimeError('object count is already computed')

anno = {}
for key, fn in tqdm(ANNO_FILE.items(), desc='reading annotations'):
    anno[key] = json.load(open(os.path.join(ANNO_DIR, fn), 'r'))

object_count = {}
for entry in tqdm(anno['objects'], desc='count object in objects'):
    for obj in entry['objects']:
        if 'name' in obj:
            name = obj['name']
            object_count[name] = object_count.get(name, 0) + 1
        if 'names' in obj:
            for name in obj['names']:
                object_count[name] = object_count.get(name, 0) + 1


for entry in tqdm(anno['attributes'], desc='count object in attributes'):
    for attr in entry['attributes']:
        if 'name' in attr:
            name = attr['name']
            object_count[name] = object_count.get(name, 0) + 1
        if 'names' in attr:
            for name in attr['names']:
                object_count[name] = object_count.get(name, 0) + 1

for entry in tqdm(anno['relationships'], desc='count object in relationships'):
    for rel in entry['relationships']:
        for key in ['object', 'subject']:
            if 'name' in rel[key]:
                name = rel[key]['name']
                object_count[name] = object_count.get(name, 0) + 1
            if 'names' in rel[key]:
                for name in rel[key]['names']:
                    object_count[name] = object_count.get(name, 0) + 1

print('save object count to: {}'.format(OUTPUT_PATH))
json.dump(object_count, open(OUTPUT_PATH, 'w'))
