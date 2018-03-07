"""
Converting densecap output bounding boxes to match original image sizes.

Run this script from root directory, not from ./data directory
"""
import argparse
import h5py
import json
import numpy as np

from tqdm import tqdm

from util import box_utils

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--densecap_output', type=str,
                    default='data/VisualGenome/VG_100K_densecap_output'
                    '/results_densecap720.json', help=' ')
parser.add_argument('--image_meta_data', type=str,
                    default='data/VisualGenome/annotations/image_data.json',
                    help=' ')
parser.add_argument('--resized_output', type=str,
                    default='data/VisualGenome/VG_100K_densecap_output'
                    '/results_original_size.hdf5', help=' ')
config = parser.parse_args()

densecap_output = json.load(open(config.densecap_output, 'r'))
image_meta_data = json.load(open(config.image_meta_data, 'r'))

id2meta_data = {i['image_id']: i for i in image_meta_data}

densecap_size = float(densecap_output['opts'][0]['image_size'])

f = h5py.File(config.resized_output, 'w')
for r in tqdm(densecap_output['results'], desc='processing results'):
    id = int(r['img_name'].replace('.jpg', ''))
    w = id2meta_data[id]['width']
    h = id2meta_data[id]['height']

    boxes = np.array(r['boxes'], dtype=np.float32)
    frac = max(w, h) / densecap_size
    new_boxes = box_utils.scale_boxes_xywh(boxes, frac)

    id_grp = f.create_group(str(id))
    id_grp['boxes'] = new_boxes
    id_grp['scores'] = np.array(r['scores'], dtype=np.float32)
f.close()
