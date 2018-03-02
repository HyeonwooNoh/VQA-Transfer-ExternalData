import argparse
import h5py
import json
import os
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--object_dataset_path', type=str,
                    default='preprocessed/objects_vocab50_min_occ20',
                    help=' ')
parser.add_argument('--attribute_dataset_path', type=str,
                    default='preprocessed/attributes_vocab50_min_occ20',
                    help=' ')
parser.add_argument('--relationship_dataset_path', type=str,
                    default='preprocessed/relationships_vocab50_min_occ20',
                    help=' ')
parser.add_argument('--region_dataset_path', type=str,
                    default='preprocessed/region_descriptions_vocab50',
                    help=' ')
parser.add_argument('--vocab_path', type=str,
                    default='preprocessed/vocab50.json', help=' ')
parser.add_argument('--save_used_wordset_path', type=str,
                    default='preprocessed/vocab50_used_wordset.hdf5', help=' ')
config = parser.parse_args()

vocab = json.load(open(config.vocab_path, 'r'))

wordset = set()
wordset.add(vocab['dict']['<s>'])
wordset.add(vocab['dict']['<e>'])
wordset.add(vocab['dict']['<unk>'])

# objects
with h5py.File(os.path.join(config.object_dataset_path, 'data.hdf5'), 'r') as f:
    intseqs = f['data_info']['objects_intseq'].value
    intseq_lens = f['data_info']['objects_intseq_len'].value
    for intseq, intseq_len in zip(intseqs, intseq_lens):
        for i in intseq[:intseq_len]:
            wordset.add(i)

# attributes
with h5py.File(os.path.join(config.attribute_dataset_path, 'data.hdf5'), 'r') as f:
    intseqs = f['data_info']['attributes_intseq'].value
    intseq_lens = f['data_info']['attributes_intseq_len'].value
    for intseq, intseq_len in zip(intseqs, intseq_lens):
        for i in intseq[:intseq_len]:
            wordset.add(i)

# relationships
with h5py.File(os.path.join(config.relationship_dataset_path, 'data.hdf5'), 'r') as f:
    intseqs = f['data_info']['relationships_intseq'].value
    intseq_lens = f['data_info']['relationships_intseq_len'].value
    for intseq, intseq_len in zip(intseqs, intseq_lens):
        for i in intseq[:intseq_len]:
            wordset.add(i)

# region_descriptions
with h5py.File(os.path.join(config.region_dataset_path, 'data.hdf5'), 'r') as f:
    ids = open(os.path.join(config.region_dataset_path, 'id.txt'),
               'r').read().splitlines()
    for id_str in tqdm(ids, desc='region_description'):
        image_id, id = id_str.split()
        entry = f[image_id][id]
        for i in entry['description'].value:
            wordset.add(i)

with h5py.File(config.save_used_wordset_path, 'w') as f:
    f['used_wordset'] = np.array(sorted(list(wordset)), dtype=np.int32)
