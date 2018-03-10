import argparse
import h5py
import json
import numpy as np

GLOVE_VOCAB_PATH = 'data/preprocessed/glove_vocab.json'
GLOVE_PARAM_PATH = 'data/preprocessed/glove.6B.300d.hdf5'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vocab_path', type=str,
                    default='data/preprocessed/new_vocab50.json', help=' ')
parser.add_argument('--filtered_glove_path', type=str,
                    default='data/preprocessed/glove.new_vocab50.300d.hdf5',
                    help=' ')
config = parser.parse_args()

vocab = json.load(open(config.vocab_path, 'r'))
assert vocab['vocab'][-1] == '<unk>', 'vocab[-1] should be <unk>'
assert vocab['vocab'][-2] == '<e>', 'vocab[-2] should be <e>'
assert vocab['vocab'][-3] == '<s>', 'vocab[-3] should be <s>'

glove_vocab = json.load(open(GLOVE_VOCAB_PATH, 'r'))

used_vocab_idx = [glove_vocab['dict'][v] for v in vocab['vocab'][:-3]]

with h5py.File(GLOVE_PARAM_PATH, 'r') as f:
    glove_param = f['param'].value

subset_param = np.take(glove_param, used_vocab_idx, axis=1)
with h5py.File(config.filtered_glove_path, 'w') as f:
    f['param'] = subset_param
print('New glove subset param is saved at: {}'.format(
    config.filtered_glove_path))
