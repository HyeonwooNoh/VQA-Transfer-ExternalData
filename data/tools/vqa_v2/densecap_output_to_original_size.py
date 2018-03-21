"""
Converting densecap output bounding boxes to match original image sizes.

Run this script from root directory, not from ./data directory
"""
import argparse
import h5py
import json
import numpy as np
import os

from tqdm import tqdm
from PIL import Image

from util import box_utils, log

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--densecap_dir', type=str,
                    default='data/VQA_v2/densecap', help=' ')
parser.add_argument('--image_dir', type=str,
                    default='data/VQA_v2/images', help=' ')
parser.add_argument('--densecap_name', type=str,
                    default='results_densecap720.json', help=' ')
parser.add_argument('--resized_name', type=str,
                    default='results_original_size.hdf5', help=' ')
config = parser.parse_args()


for split in ['train2014', 'val2014', 'test2015']:
    densecap_path = os.path.join(config.densecap_dir, split,
                                 config.densecap_name)
    log.warn('loading densecap: {}'.format(densecap_path))
    densecap = json.load(open(densecap_path, 'r'))
    densecap_size = float(densecap['opts'][0]['image_size'])

    resized_path = os.path.join(config.densecap_dir, split,
                                config.resized_name)
    log.warn('create resized densecap file: {}'.format(resized_path))
    f = h5py.File(resized_path, 'w')
    for r in tqdm(densecap['results'], desc='processing {}'.format(split)):
        image_path = os.path.join(split, r['img_name'])
        image = Image.open(os.path.join(config.image_dir, image_path))
        w, h = image.size

        boxes = np.array(r['boxes'], dtype=np.float32)
        frac = max(w, h) / densecap_size
        new_boxes = box_utils.scale_boxes_xywh(boxes, frac)

        id_grp = f.create_group(image_path.replace('/', '-'))
        id_grp['boxes'] = new_boxes
        id_grp['scores'] = np.array(r['scores'], dtype=np.float32)
    f.close()
    log.info('processing is done: {}'.format(resized_path))
