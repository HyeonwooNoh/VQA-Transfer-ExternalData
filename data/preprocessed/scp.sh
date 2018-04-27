#!/bin/bash
download() {
  filename=$(basename $1)
  if [ ! -e $filename ]; then
    if [[ $filename = *"."* ]]; then
      scp -P 7777 `whoami`@147.47.209.134:/home/hyeonwoonoh/vlmap/$1 .
    else
      scp -P 7777 -r `whoami`@147.47.209.134:/home/hyeonwoonoh/vlmap/$1 .
    fi
  fi
}

download "data/preprocessed/glove.new_vocab50.300d.hdf5"
download "data/preprocessed/new_vocab50.json"
download "data/preprocessed/glove.6B.300d.hdf5"
download "data/preprocessed/glove_vocab.json"
download "data/preprocessed/visualgenome/memft_all_new_vocab50_obj3000_attr1000_maxlen10"
download "data/preprocessed/vqa_v2/qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1"
download "data/preprocessed/enwiki/enwiki_processed"
