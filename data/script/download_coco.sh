#!/bin/bash
mkdir COCO
cd COCO

wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget https://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip

# https://www.dropbox.com/s/1t9nrbevzqn93to/coco.tar.gz?dl=0
# http://cocodataset.org/#download: 2014 training, 2014 val

unzip caption_datasets.zip
unzip annotations_trainval2014.zip
unzip stanford-corenlp-full-2017-06-09.zip

#python prepro/prepro_dic_coco.py --input_json data/coco/dataset_coco.json --split normal --output_dic_json data/coco/dic_coco.json --output_cap_json data/coco/cap_coco.json

