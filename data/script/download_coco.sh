#!/bin/bash
mkdir COCO
cd COCO

download_unzip_rm() {
    url=$1
    filename=$(basename $1)

    # 1. download
    if [ -z "$2" ]; then
        wget $url
    else
        wget $url -O $2
        filename=$2
    fi

    # 2. unzip
    if [[ $filename == *.tar.gz ]]; then
        tar -zxvf $filename
    else
        unzip $filename
    fi
    rm $filename
}

download_unzip_rm http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
download_unzip_rm http://images.cocodataset.org/annotations/annotations_trainval2014.zip

download_unzip_rm "https://www.dropbox.com/s/2gzo4ops5gbjx5h/coco_detection.h5.tar.gz?dl=1" coco_detection.h5.tar.gz
#download_unzip_rm "https://www.dropbox.com/s/1t9nrbevzqn93to/coco.tar.gz?dl=1" coco.tar.gz
#download_unzip_rm "https://www.dropbox.com/s/tevyub9rxz6d22l/coco_robust.tar.gz?dl=1" coco_robust.tar.gz
download_unzip_rm "https://www.dropbox.com/s/tevyub9rxz6d22l/coco_noc.tar.gz?dl=1" coco_robust.tar.gz

wget "https://raw.githubusercontent.com/jiasenlu/NeuralBabyTalk/master/data/noc_coco/split_noc_coco.json"
