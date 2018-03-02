# Data
Directory containing datasets

### Preparation
1. Download [VisualGenome version 1.4](http://visualgenome.org/api/v0/api_home.html)
1. Download [GloVe word vector](https://nlp.stanford.edu/projects/glove/)
1. Download nets (pre-trained cnn weights)

You can use the following scripts:
- script/download\_visualgenome.sh
- script/download\_glove.sh
- script/download\_nets.sh

### Structure (After downloading)
- VisualGenome/VG\_100K/\*.jpg
- VisualGenome/annotations/objects.json
- VisualGenome/annotations/attributes.json
- VisualGenome/annotations/relationships.json
- VisualGenome/annotations/region\_descriptions.json
- VisualGenome/annotations/region\_graphs.json
- VisualGenome/annotations/scene\_graphs.json
- VisualGenome/annotations/object\_alias.txt
- VisualGenome/annotations/relationship\_alias.txt
- GloVe/glove.6B.50d.txt
- GloVe/glove.6B.100d.txt
- GloVe/glove.6B.200d.txt
- GloVe/glove.6B.300d.txt

### Data preprocessing.
Use following commands.

1. construct glove vocab and image\_scplit
```python
python tools/construct_vocab.py
python tools/construct_image_split.py
```
1. construct dataset with glove vocab
```python
python tools/generator_objects.py --vocab_path preprocessed/vocab.json
python tools/generator_attributes.py --vocab_path preprocessed/vocab.json
python tools/generator_relationships.py --vocab_path preprocessed/vocab.json
python tools/generator_region_descriptions.py --vocab_path preprocessed/vocab.json
```
1. extract frequent vocab from constructed datasets
```python
python tools/construct_frequent_vocab.py --min_occurrence 50
```
1. construct dataset with new vocab
```python
python tools/generator_objects.py --vocab_path preprocessed/vocab50.json
python tools/generator_attributes.py --vocab_path preprocessed/vocab50.json
python tools/generator_relationships.py --vocab_path preprocessed/vocab50.json
python tools/generator_region_descriptions.py --vocab_path preprocessed/vocab50.json
```
1. construct used wordset
```python
python tools/construct_used_wordset.py --vocab_path preprocessed/vocab50.json\
--save_used_wordset_path preprocessed/vocab50_used_wordset.hdf5
```
