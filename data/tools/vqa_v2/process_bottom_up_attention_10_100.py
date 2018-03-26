import argparse
import base64
import csv
import h5py
import json
import os
import sys
import numpy as np

from tqdm import tqdm

from util import box_utils, log


BOTTOM_UP_FILE_NAMES = [
    'trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv',
    'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0',
    'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1',
    'trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',
]

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--qa_split_dir', type=str,
                    default='data/preprocessed/vqa_v2'
                    '/qa_split_thres1_500_thres2_50', help=' ')
parser.add_argument('--save_used_vfeat_name', type=str,
                    default='used_vfeat_bottom_up_10_100.hdf5', help=' ')
parser.add_argument('--bottom_up_dir', type=str,
                    default='data/VQA_v2/bottom_up_attention_10_100', help=' ')
config = parser.parse_args()

config.vfeat_path = os.path.join(config.qa_split_dir,
                                 config.save_used_vfeat_name)
if os.path.exists(config.vfeat_path):
    raise ValueError('The file exists. Do not overwrite: {}'.format(
        config.vfeat_path))
config.anno_path = os.path.join(config.qa_split_dir, 'merged_annotations.json')
qid2anno = json.load(open(config.anno_path, 'r'))
log.infov('processing anno')
image_id2path = {}
for anno in qid2anno.values():
    image_id2path[anno['image_id']] = anno['image_path']
log.infov('processing anno is done')

csv.field_size_limit(sys.maxsize)

int_field = ['image_id', 'image_w', 'image_h', 'num_boxes']
np_field = ['boxes', 'features']

f = h5py.File(config.vfeat_path, 'w')
max_box_num = 0
for file_name in BOTTOM_UP_FILE_NAMES:
    log.warn('process: {}'.format(file_name))

    tsv_in_file = open(os.path.join(config.bottom_up_dir, file_name), 'r+b')

    reader = csv.DictReader(tsv_in_file, delimiter='\t',
                            fieldnames=(int_field + np_field))
    for item in tqdm(reader, desc='processing reader'):
        for field in int_field:
            item[field] = int(item[field])
        for field in np_field:
            item[field] = np.frombuffer(base64.decodestring(item[field]),
                                        dtype=np.float32).reshape(
                                            (item['num_boxes'], -1))
        image_id = item['image_id']
        image_path = image_id3path[image_id]
        image_path_id = image_path.replace('/', '-')

        grp = f.create_group(image_path_id)
        grp['image_num_id'] = image_id
        grp['original_image_w'] = item['image_w']
        grp['original_image_h'] = item['image_h']
        grp['box_image_w'] = 540
        grp['box_image_h'] = 540
        grp['num_box'] = item['num_boxes']
        grp['vfeat'] = item['features'].astype(np.float32)

        WIDTH = 540.0
        HEIGHT = 540.0

        frac_x = WIDTH / float(item['image_w'])
        frac_y = HEIGHT / float(item['image_h'])
        box_x1y1x2y2 = item['boxes'].astype(np.float32)
        box = box_utils.scale_boxes_x1y1x2y2(box_x1y1x2y2, [frac_x, frac_y])
        grp['box'] = box
        grp['normal_box'] = box_utils.normalize_boxes_x1y1x2y2(
            box, WIDTH, HEIGHT)
        max_box_num = max(max_box_num, item['num_boxes'])
data_info = f.create_group('data_info')
data_info['max_box_num'] = max_box_num
data_info['pretrained_param_path'] = 'bottom_up_attention_10_100'
f.close()
log.warn('done')
