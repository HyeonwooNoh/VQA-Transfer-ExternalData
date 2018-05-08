import os
import time
import shlex
import subprocess
from glob import glob
from datetime import datetime
from collections import defaultdict

from util import log

def list_dir(directory, prefix="", postfix=""):
    lists = glob(directory + "/{}*{}".format(prefix, postfix))
    lists.sort()
    return lists

def parallel_run(commands, config):
    for idx, cmd in enumerate(commands):
        num = idx % config.num_thread
        cmd = 'CUDA_VISIBLE_DEVICES={} '.format(num) + cmd

        if num != config.num_thread - 1:
            run(cmd, config, wait=False)
        else:
            run(cmd, config)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def symlink(src_path, dst_path):
    root_dir = os.path.dirname(os.path.realpath(__file__))

    src_path = os.path.join(root_dir, src_path)
    dst_path = os.path.join(root_dir, dst_path)

    if os.path.exists(dst_path):
        log.info("{} already exists".format(dst_path))
    else:
        log.info("Sym link: {}->{}".format(src_path, dst_path))
        os.symlink(src_path, dst_path)

def run(cmd, config, wait=True):
    print(" [*] {}".format(cmd))
    if not config.debug:
        if wait:
            subprocess.call(cmd, shell=True)
        else:
            subprocess.Popen(cmd, shell=True)
        time.sleep(1.1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug', type=int, default=0, help='0: normal, 1: debug')
    parser.add_argument('--num_thread', type=int, default=2)
    parser.add_argument('--time_str', type=str, default=None)
    parser.add_argument('--skip_vqa', type=int, default=0)
    parser.add_argument('--skip_vlmap', type=int, default=0)
    parser.add_argument('--result_dir', type=str, default='experiments/important')

    config = parser.parse_args()

    time_str = config.time_str or datetime.now().strftime("%Y%m%d-%H%M%S")

    TAG = time_str + "_vqa_bf_or_ws_123_from_vlamp_234_345_456_depth{}"
    VLMAP_BASE = "vlmap_bf_or_wordset_withatt_sp_d_memft_all_new_vocab50_obj3000_attr1000_maxlen10_expand_depth_bs512_lr0.001_dp{}_seed{}_*"

    VLMAP_SEEDS = [234, 345, 456]
    VQA_SEEDS = [123, 234, 345]
    DEPTHS = ['True', 'False']
    
    #########################
    # 1. find_word_group.py
    #########################

    cmds = []
    for depth in DEPTHS:
        cmd = 'python data/tools/visualgenome/find_word_group.py --expand_depth={}'.format(depth)
        cmds.append(cmd)
        #run(cmd, config)

    parallel_run(cmds, config)
    
    ###############################
    # 2. vlmap_memft/trainer.py
    ###############################

    if not config.skip_vlmap:
        cmds = []
        for depth in DEPTHS:
            for vlmap_seed in VLMAP_SEEDS:
                cmd = 'python vlmap_memft/trainer.py --model_type=vlmap_bf_or_wordset_withatt_sp --prefix=expand_depth --max_train_iter=4810 --seed={} --expand_depth={}'.format(vlmap_seed, depth)
                cmds.append(cmd)

        parallel_run(cmds, config)
    
    #########################################
    # 3. symlink to experiments/important/*
    #########################################

    important_dirs = {}
    important_sub_dirs = defaultdict(list)

    for depth in DEPTHS:
        tag = TAG.format(depth)
        base_path = "{}/{}".format(config.result_dir, tag)
        makedirs(base_path)

        important_dirs[depth] = base_path

        dirs = list_dir("train_dir", prefix=VLMAP_BASE.format("*", "*"))
        for directory in dirs:
            dir_time_str = directory.rsplit('_', 1)[1]

            if time_str < dir_time_str:
                src_path = directory
                dst_path = os.path.join(base_path, directory.rsplit('/')[-1].rsplit('_', 1)[0])

                symlink(src_path, dst_path)
                important_sub_dirs[depth].append(dst_path)
    
    #########################################
    # 4. vlmap_memft/export_word_weights.py
    #########################################

    cmds = []
    for key in important_sub_dirs:
        for directory in important_sub_dirs[key]:
            checkpoint = os.path.join(directory, "model-4800")
            cmd = "python vlmap_memft/export_word_weights.py --checkpoint={}".format(checkpoint)
            cmds.append(cmd)

    parallel_run(cmds, config)

    #########################################
    # 5. vqa/trainer.py
    #########################################

    def find_arg(path, arg):
        for item in path.split('_')[::-1]:
            if item.startswith(arg):
                value = item[len(arg):]
                return value

    if not config.skip_vqa:
        cmds = []
        for key in important_dirs:
            for path in important_sub_dirs[key]:
                for vqa_seed in VQA_SEEDS:
                    dp = find_arg(path, "dp")
                    vlmap_seed = find_arg(path, "seed")

                    cmd = "python vqa/trainer.py" \
                        " --pretrained_param_path {path}/model-4800" \
                        " --vlmap_word_weight_dir {path}/word_weights_model-4800" \
                        " --prefix dp{dp}_sd{vlmap_seed}vqa_sd{vqa_seed} --seed {vqa_seed}" \
                        .format(path=path, dp=dp, vlmap_seed=vlmap_seed, vqa_seed=vqa_seed)
                    cmds.append(cmd)

        parallel_run(cmds, config)

    #########################################
    # -2. vqa/eval_multiple_model.py
    #########################################

    cmds = []
    for key, important_dir in important_dirs.items():
        cmd = "python vqa/eval_multiple_model.py --root_train_dir={}".format(important_dir)
        cmds.append(cmd)

    parallel_run(cmds, config)

    #########################################
    # -1. vqa/eval_collection.py
    #########################################

    cmds = []
    for key, important_dir in important_dirs.items():
        cmd = "python vqa/eval_collection.py --root_train_dir={}".format(important_dir)
        cmds.append(cmd)

    parallel_run(cmds, config)
