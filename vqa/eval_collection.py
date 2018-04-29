import argparse
import cPickle
import glob
import os

from tqdm import tqdm

from util import log

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--root_train_dir', type=str, default=None, help=' ')
parser.add_argument('--train_dirs', nargs='+', type=str, default=[], help=' ')
parser.add_argument('--split', type=str, default='test', help=' ',
                    choices=['train', 'val', 'testval', 'test'])
config = parser.parse_args()

if config.root_train_dir is None and len(config.train_dirs) == 0:
    raise ValueError('Set either root_train_dir or train_dirs')
if config.root_train_dir is not None and len(config.train_dirs) > 0:
    raise ValueError('Do not set both root_train_dir and train_dirs')

if config.root_train_dir is None:
    all_train_dirs = config.train_dirs
else:
    all_train_dirs = glob.glob(os.path.join(config.root_train_dir, 'vqa_*'))
all_train_dirs = sorted(all_train_dirs)

log.warn('all_train_dirs:')
for i, train_dir in enumerate(all_train_dirs):
    log.infov('{:02d}: {}'.format(i, train_dir))

for i_train_dir, train_dir in enumerate(all_train_dirs):
    log.warn('[{:02d}] train_dir: {}'.format(i_train_dir, train_dir))

    eval_dirs = glob.glob(os.path.join(
        train_dir, 'model-*_eval_{}_*'.format(config.split)))
    eval_iter2dir = {int(e.split('model-')[1].split('_eval')[0]): e
                     for e in eval_dirs}
    iters = sorted(eval_iter2dir)

    collect_results = {
        'iter': [],
        'testonly_score': [],
        'testonly_score_num_point': [],
        'test_obj_only_score': [],
        'test_obj_only_score_num_point': [],
        'test_attr_only_score': [],
        'test_attr_only_score_num_point': [],
    }
    collect_list = [('iter', 'testonly_score', 'testonly_score_num_point',
                     'test_obj_only_score', 'test_obj_only_score_num_point',
                     'test_attr_only_score', 'test_attr_only_score_num_point')]
    for i in tqdm(iters, desc='iters'):
        eval_dir = eval_iter2dir[i]
        results = cPickle.load(open(os.path.join(eval_dir, 'results.pkl'), 'rb'))
        avg = results['avg_eval_report']
        collect_results['iter'].append(i)
        collect_results['testonly_score'].append(avg['testonly_score'])
        collect_results['testonly_score_num_point'].append(avg['testonly_score_num_point'])
        collect_results['test_obj_only_score'].append(
            avg['test_obj_only_score'])
        collect_results['test_obj_only_score_num_point'].append(
            avg['test_obj_only_score_num_point'])
        collect_results['test_attr_only_score'].append(
            avg['test_attr_only_score'])
        collect_results['test_attr_only_score_num_point'].append(
            avg['test_attr_only_score_num_point'])
        collect_list.append(
            ('{:05d}'.format(i), '{:.5f}'.format(avg['testonly_score']),
             '{:08d}'.format(avg['testonly_score_num_point']),
             '{:.5f}'.format(avg['test_obj_only_score']),
             '{:08d}'.format(avg['test_obj_only_score_num_point']),
             '{:.5f}'.format(avg['test_attr_only_score']),
             '{:08d}'.format(avg['test_attr_only_score_num_point'])))

    collect_txt_path = os.path.join(
        train_dir, 'collect_eval_{}_result.txt'.format(config.split))
    collect_pkl_path = os.path.join(
        train_dir, 'collect_eval_{}_result.pkl'.format(config.split))

    with open(collect_txt_path, 'w') as f:
        for collect in tqdm(collect_list, desc='write collect_txt'):
            f.write(' '.join(collect) + '\n')

    log.warn('result is saved in {}'.format(collect_txt_path))

    cPickle.dump(collect_results, open(collect_pkl_path, 'wb'))
    log.warn('result is saved in {}'.format(collect_pkl_path))
