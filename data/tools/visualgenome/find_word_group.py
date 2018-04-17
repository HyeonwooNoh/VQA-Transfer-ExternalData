import argparse
import cPickle
import json
import os

from nltk.corpus import wordnet as wn
from textblob import Word
from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--genome_annotation_dir', type=str,
                    default='data/VisualGenome/annotations', help=' ')
parser.add_argument('--dir_name', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
parser.add_argument('--min_num_word', type=int, default=5, help='min num word in set')
config = parser.parse_args()

config.object_synset_path = os.path.join(
    config.genome_annotation_dir, 'object_synsets.json')
config.attribute_synset_path = os.path.join(
    config.genome_annotation_dir, 'attribute_synsets.json')
config.answer_dict_path = os.path.join(
    config.dir_name, 'answer_dict.pkl')
config.save_wordset_path = os.path.join(
    config.dir_name, 'wordset_dict{}.pkl'.format(config.min_num_word))

object_synsets = json.load(open(config.object_synset_path, 'r'))
attribute_synsets = json.load(open(config.attribute_synset_path, 'r'))

synsets = {}
for key, val in object_synsets.items():
    synsets[key] = val
for key, val in attribute_synsets.items():
    if key not in synsets:
        synsets[key] = val

answer_dict = cPickle.load(open(config.answer_dict_path, 'rb'))


vocab_with_synset = []
for v in answer_dict['vocab']:
    v_w = Word('_'.join(v.split()))
    if len(v_w.synsets) > 0:
        synsets[v] = v_w.synsets[0].name()
    if v in synsets:
        vocab_with_synset.append(v)

hypernym_set = set()
vocab_hypernyms = {}
for v in tqdm(vocab_with_synset, desc='make hypernymset'):
    word_synset = wn.synset(synsets[v])
    hypernyms = set([k[0] for k in list(word_synset.hypernym_distances())])
    hypernyms = hypernyms - set([word_synset])
    hypernym_set = hypernym_set | hypernyms
    vocab_hypernyms[v] = [h.name() for h in list(hypernyms)]

hypernym_vocab = [v.name() for v in list(hypernym_set)]
hypernym_dict = {v: i for i, v in enumerate(hypernym_vocab)}

hypernym_wordset = {v: [] for v in hypernym_vocab}
for v, v_hyper in tqdm(vocab_hypernyms.items(), desc='find wordset'):
    for h in v_hyper:
        hypernym_wordset[h].append(v)

for k, v in hypernym_wordset.items():
    if len(v) < config.min_num_word:
        del hypernym_wordset[k]

# add wordset with all answers
hypernym_wordset['all_answers'] = answer_dict['vocab']

wordset_dict = {}
wordset_dict['vocab'] = hypernym_wordset.keys()
wordset_dict['dict'] = {v: i for i, v in enumerate(wordset_dict['vocab'])}

wordset_dict['ans2wordset'] = {answer_dict['dict'][v]: []
                               for v in answer_dict['vocab']}
wordset_dict['wordset2ans'] = {wordset_dict['dict'][k]: set()
                               for k in wordset_dict['vocab']}
for k, v_list in hypernym_wordset.items():
    k_idx = wordset_dict['dict'][k]
    for v in v_list:
        v_idx = answer_dict['dict'][v]
        wordset_dict['ans2wordset'][v_idx].append(k_idx)
        wordset_dict['wordset2ans'][k_idx].add(v_idx)

cPickle.dump(wordset_dict, open(config.save_wordset_path, 'wb'))
print('wordset is saved in : {}'.format(config.save_wordset_path))
