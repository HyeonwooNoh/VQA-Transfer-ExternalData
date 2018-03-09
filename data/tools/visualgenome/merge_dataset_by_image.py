import argparse
import h5py
import json
import os
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--objects_dir', type=str,
                    default='preprocessed/visualgenome'
                    '/objects_new_vocab50_min_occ20',
                    help=' ')
parser.add_argument('--attributes_dir', type=str,
                    default='preprocessed/visualgenome'
                    '/attributes_new_vocab50_min_occ20',
                    help=' ')
parser.add_argument('--relationships_dir', type=str,
                    default='preprocessed/visualgenome'
                    '/relationships_new_vocab50_min_occ20',
                    help=' ')
parser.add_argument('--regions_dir', type=str,
                    default='preprocessed/visualgenome'
                    '/region_descriptions_new_vocab50_min_len10',
                    help=' ')
parser.add_argument('--vocab_path', type=str,
                    default='preprocessed/new_vocab50.json', help=' ')
parser.add_argument('--image_split_path', type=str,
                    default='preprocessed/visualgenome/image_split.json',
                    help=' ')
parser.add_argument('--merged_dataset_dir',
                    default='preprocessed/visualgenome'
                    '/merged_by_image_new_vocab50',
                    help=' ')
config = parser.parse_args()


if not os.path.exists(config.merged_dataset_dir):
    print('Create dataset dir: {}'.format(config.merged_dataset_dir))
    os.makedirs(config.merged_dataset_dir)
else:
    raise ValueError('The directory {} already exists. Do not overwrite'.format(
        config.merged_dataset_dir))


def construct_image2id(id_pairs):
    image2id = {}
    for id_pair in id_pairs:
        image_id, id = id_pair.split()
        if image_id not in image2id:
            image2id[image_id] = []
        image2id[image_id].append(id)
    return image2id

print('Constructing image2id mapping')
object_image2id = construct_image2id(
    open(os.path.join(config.objects_dir, 'id.txt'), 'r').read().splitlines())
attribute_image2id = construct_image2id(
    open(os.path.join(config.attributes_dir, 'id.txt'), 'r').read().splitlines())
relationship_image2id = construct_image2id(
    open(os.path.join(config.relationships_dir, 'id.txt'), 'r').read().splitlines())
region_image2id = construct_image2id(
    open(os.path.join(config.regions_dir, 'id.txt'), 'r').read().splitlines())
print('Done')

print('Loading separate datasets..')
object_f = h5py.File(os.path.join(config.objects_dir, 'data.hdf5'), 'r')
attr_f = h5py.File(os.path.join(config.attributes_dir, 'data.hdf5'), 'r')
relation_f = h5py.File(os.path.join(config.relationships_dir, 'data.hdf5'), 'r')
region_f = h5py.File(os.path.join(config.regions_dir, 'data.hdf5'), 'r')
print('Done')

f = h5py.File(os.path.join(config.merged_dataset_dir, 'data.hdf5'), 'w')
data_info = f.create_group('data_info')
# objects
data_info['max_object_name_len'] = object_f['data_info']['max_name_length'].value
data_info['max_object_num_per_box'] = \
    object_f['data_info']['max_num_names'].value
data_info['min_object_occurrence'] = \
    object_f['data_info']['min_occurrence'].value
data_info['num_unique_objects'] = \
    object_f['data_info']['num_unique_objects'].value
data_info['objects_intseq'] = object_f['data_info']['objects_intseq'].value
data_info['objects_intseq_len'] = \
    object_f['data_info']['objects_intseq_len'].value
data_info['total_num_objects'] = object_f['data_info']['num_data'].value
data_info['total_num_objects_train'] = object_f['data_info']['num_train'].value
data_info['total_num_objects_test'] = object_f['data_info']['num_test'].value
data_info['total_num_objects_val'] = object_f['data_info']['num_val'].value
# attributes
data_info['max_attribute_name_len'] = \
    attr_f['data_info']['max_name_length'].value
data_info['max_attribute_num_per_box'] = \
    attr_f['data_info']['max_num_names'].value
data_info['min_attribute_occurrence'] = \
    attr_f['data_info']['min_occurrence'].value
data_info['num_unique_attributes'] = \
    attr_f['data_info']['num_unique_attributes'].value
data_info['attributes_intseq'] = attr_f['data_info']['attributes_intseq'].value
data_info['attributes_intseq_len'] = \
    attr_f['data_info']['attributes_intseq_len'].value
data_info['total_num_attributes'] = attr_f['data_info']['num_data'].value
data_info['total_num_attributes_train'] = attr_f['data_info']['num_train'].value
data_info['total_num_attributes_test'] = attr_f['data_info']['num_test'].value
data_info['total_num_attributes_val'] = attr_f['data_info']['num_val'].value
# relationships
data_info['max_relationship_name_len'] = \
    relation_f['data_info']['max_name_length'].value
data_info['max_relationship_num_per_box'] = \
    relation_f['data_info']['max_num_names'].value
data_info['min_relationship_occurrence'] = \
    relation_f['data_info']['min_occurrence'].value
data_info['num_unique_relationships'] = \
    relation_f['data_info']['num_unique_relationships'].value
data_info['relationships_intseq'] = \
    relation_f['data_info']['relationships_intseq'].value
data_info['relationships_intseq_len'] = \
    relation_f['data_info']['relationships_intseq_len'].value
data_info['total_num_relationships'] = relation_f['data_info']['num_data'].value
data_info['total_num_relationships_train'] = relation_f['data_info']['num_train'].value
data_info['total_num_relationships_test'] = relation_f['data_info']['num_test'].value
data_info['total_num_relationships_val'] = relation_f['data_info']['num_val'].value
# region descriptions
data_info['max_description_length'] = \
    region_f['data_info']['max_length'].value
data_info['num_unique_descriptions'] = \
    region_f['data_info']['num_unique_descriptions'].value
data_info['total_num_descriptions'] = region_f['data_info']['num_data'].value
data_info['total_num_descriptions_train'] = region_f['data_info']['num_train'].value
data_info['total_num_descriptions_test'] = region_f['data_info']['num_test'].value
data_info['total_num_descriptions_val'] = region_f['data_info']['num_val'].value

print('Load split: {}'.format(config.image_split_path))
image_split = json.load(open(config.image_split_path, 'r'))
image_ids = image_split['train'] + image_split['test'] + image_split['val']
train_image_set = set(image_split['train'])
test_image_set = set(image_split['test'])
val_image_set = set(image_split['val'])


train_ids = []
test_ids = []
val_ids = []
min_box = {'obj': 1000, 'attr': 1000, 'rel': 1000, 'region': 1000}
max_box = {'obj': 0, 'attr': 0, 'rel': 0, 'region': 0}
for image_id in tqdm(image_ids, desc='process_image_ids'):
    image_id = str(image_id)
    ids = {}
    ids['obj'] = object_image2id.get(image_id, [])
    ids['attr'] = attribute_image2id.get(image_id, [])
    ids['rel'] = relationship_image2id.get(image_id, [])
    ids['region'] = region_image2id.get(image_id, [])
    if len(ids['obj']) == 0 or len(ids['attr']) == 0 or \
            len(ids['rel']) == 0 or len(ids['region']) == 0:
        continue
    for key in ids.keys():
        min_box[key] = min(min_box[key], len(ids[key]))
        max_box[key] = max(max_box[key], len(ids[key]))

    entry = {}
    entry['obj'] = object_f[image_id]
    entry['attr'] = attr_f[image_id]
    entry['rel'] = relation_f[image_id]
    entry['region'] = region_f[image_id]

    boxes_in_image = {}  # [#entry, 4 (x, y, w, h)]
    name_ids_in_image = {}  # [#entry, max_num_names]
    name_len_in_image = {}  # [#entry, max_num_names]
    names_in_image = {}  # [#entry, max_num_names, max_name_len]
    num_names_in_image = {}  # [#entry]
    for key in ['obj', 'attr', 'rel']:
        boxes_in_image[key] = []
        name_ids_in_image[key] = []
        name_len_in_image[key] = []
        names_in_image[key] = []
        num_names_in_image[key] = []

        max_name_len = max([entry[key][id]['names'].value.shape[1]
                            for id in ids[key]])
        max_num_names = max([entry[key][id]['names'].value.shape[0]
                             for id in ids[key]])
        for id in ids[key]:
            e = entry[key][id]
            names = e['names'].value
            name_len = e['name_len'].value
            name_ids = e['name_ids'].value

            xywh = np.array(
                [e['x'].value, e['y'].value, e['w'].value, e['h'].value],
                dtype=np.int32)
            pad_names = np.zeros([max_num_names, max_name_len], dtype=np.int32)
            pad_names[:names.shape[0], :names.shape[1]] = names
            pad_name_len = np.zeros([max_num_names], dtype=np.int32)
            pad_name_len[:name_len.shape[0]] = name_len
            pad_name_ids = np.zeros([max_num_names], dtype=np.int32)
            pad_name_ids[:name_ids.shape[0]] = name_ids

            boxes_in_image[key].append(xywh)
            name_ids_in_image[key].append(pad_name_ids)
            name_len_in_image[key].append(pad_name_len)
            names_in_image[key].append(pad_names)
            num_names_in_image[key].append(name_len.shape[0])
        boxes_in_image[key] = np.stack(boxes_in_image[key], axis=0)
        name_ids_in_image[key] = np.stack(name_ids_in_image[key], axis=0)
        name_len_in_image[key] = np.stack(name_len_in_image[key], axis=0)
        names_in_image[key] = np.stack(names_in_image[key], axis=0)
        num_names_in_image[key] = np.array(num_names_in_image[key], dtype=np.int32)

    region_boxes = []
    region_descriptions = []
    region_description_len = []
    max_len = 0
    for id in ids['region']:
        e = entry['region'][id]
        description = e['description'].value
        xywh = np.array([e['x'].value, e['y'].value, e['w'].value, e['h'].value],
                        dtype=np.int32)
        region_boxes.append(xywh)
        region_descriptions.append(description)
        region_description_len.append(len(description))
        max_len = max(max_len, len(description))
    for i in range(len(region_descriptions)):
        description = region_descriptions[i]
        pad_description = np.zeros([max_len], dtype=np.int32)
        pad_description[:len(description)] = description
        region_descriptions[i] = pad_description
    region_boxes = np.stack(region_boxes, axis=0)
    region_descriptions = np.stack(region_descriptions, axis=0)
    region_description_len = np.array(region_description_len, dtype=np.int32)

    image_grp = f.create_group(image_id)
    prefix = {'obj': 'object', 'attr': 'attribute', 'rel': 'relationship'}
    for key in ['obj', 'attr', 'rel']:
        image_grp['{}_xywh'.format(prefix[key])] = boxes_in_image[key]
        image_grp['{}_name_ids'.format(prefix[key])] =\
            name_ids_in_image[key]
        image_grp['{}_name_len'.format(prefix[key])] =\
            name_len_in_image[key]
        # TODO(hyeonwoonoh): save names in to "{}_num_names" and save names to
        # "{}_names" and change reading code in vlmap/datasets/dataset_vlmap
        image_grp['{}_names'.format(prefix[key])] =\
            names_in_image[key]
        image_grp['{}_num_names'.format(prefix[key])] =\
            num_names_in_image[key]
    image_grp['region_xywh'] = region_boxes
    image_grp['region_descriptions'] = region_descriptions
    image_grp['region_description_len'] = region_description_len

    if int(image_id) in train_image_set: train_ids.append(image_id)
    if int(image_id) in test_image_set: test_ids.append(image_id)
    if int(image_id) in val_image_set: val_ids.append(image_id)

idf = open(os.path.join(config.merged_dataset_dir, 'id.txt'), 'w')
for image_id in train_ids + test_ids + val_ids:
    idf.write(str(image_id) + '\n')
idf.close()

data_info['num_train'] = len(train_ids)
data_info['num_test'] = len(test_ids)
data_info['num_val'] = len(val_ids)
data_info['num_data'] = len(train_ids) + len(test_ids) + len(val_ids)
prefix = {'obj': 'object', 'attr': 'attribute', 'rel': 'relationship',
          'region': 'region'}
for key in ['obj', 'attr', 'rel', 'region']:
    data_info['{}_min_num_box_in_image'.format(prefix[key])] = min_box[key]
    data_info['{}_max_num_box_in_image'.format(prefix[key])] = max_box[key]
f.close()

print('Merged dataset is created: {}'.format(config.merged_dataset_dir))
