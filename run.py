import os
import time
import shlex
import itertools
import subprocess
from glob import glob
from datetime import datetime
from collections import defaultdict

from util import log

def list_dir(directory, prefix="", postfix=""):
    lists = glob(directory + "/{}*{}".format(prefix, postfix))
    lists.sort()
    return lists

def grouper(iterable, n):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

def parallel_run(commands, config):
    groups = list(grouper(commands, config.num_thread))

    for idx, cmds in enumerate(groups):
        procs = []

        for num, cmd in enumerate(cmds):
            print(" [*] Group {}/{}, Thread {}/{}". \
                format(idx, len(groups), num, len(cmds)))

            cmd = 'CUDA_VISIBLE_DEVICES={} '.format(num % config.num_gpu) + cmd
            proc = run(cmd, config)
            procs.append(proc)

        for proc in procs:
            proc.wait()

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

def run(cmd, config):
    print(" [*] {}".format(cmd))
    if not config.debug:
        proc = subprocess.Popen(cmd, shell=True)
        time.sleep(1)
    return proc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('num_thread', type=int)
    parser.add_argument('num_gpu', type=int)
    parser.add_argument('vlmap_prefix', type=str)
    parser.add_argument('--debug', type=int, default=0, help='0: normal, 1: debug')
    parser.add_argument('--time_str', type=str, default=None)
    parser.add_argument('--enwiki_sep_num', type=int, default=4)
    parser.add_argument('--process_enwiki', type=int, default=0)
    parser.add_argument('--process_depth', type=int, default=0)
    parser.add_argument('--skip_vqa', type=int, default=0)
    parser.add_argument('--skip_vlmap', type=int, default=0)
    parser.add_argument('--result_dir', type=str, default='experiments/important')

    config = parser.parse_args()

    time_str = config.time_str or datetime.now().strftime("%Y%m%d-%H%M%S")

    vlmap_prefix = config.vlmap_prefix
    TAG = time_str + "_vqa_bf_or_ws_123_from_vlamp_234_345_456"
    VLMAP_BASE = "{vlmap_model}_d_memft_all_new_vocab50_obj3000_attr1000_maxlen10_" \
                 "{vlmap_prefix}_bs512_lr0.001_dp{depth}_seed{seed}_*"

    #VLMAP_SEEDS = [234, 345, 456]
    #VQA_SEEDS = [123, 234, 345]
    VLMAP_SEEDS = [234]
    VQA_SEEDS = [123]

    DEPTHS = ['False']
    VLMAP_MODELS = ['vlmap_bf_or_wordset_withatt_sp',
                    'vlmap_enwiki_withatt_sp']
    enwiki_preprocessing = False 
    # standard_word2vec: 3, vlmap_answer: 6
    MODEL_TYPES = ['standard_word2vec', 'vlmap_answer']
    
    #########################
    # 1. find_word_group.py
    #########################

    if config.process_depth:
        cmds = []
        for depth in DEPTHS:
            cmd = 'python data/tools/visualgenome/find_word_group.py --expand_depth={}'.format(depth)
            cmds.append(cmd)
            #run(cmd, config)

        parallel_run(cmds, config)

    if config.process_enwiki:
        cmds = []
        base_cmd = 'python data/tools/enwiki/{} --enwiki_dir=data/preprocessed/enwiki/enwiki_processed_{}_{}'

        def loop(filename, args):
            for arg, values in args.items():
                for value in values:
                    for idx in range(1, config.enwiki_sep_num+1):
                        cmd = base_cmd.format(filename, idx, config.enwiki_sep_num)
                        cmd += " --{}={}".format(arg, value)
                        cmds.append(cmd)

        loop('2_word2contexts.py', {'preprocessing': [0, 1]})

        #cmds.append(base_cmd.format('3_make_wordset.py', idx, config.enwiki_sep_num))
        parallel_run(cmds, config)
    
    ###############################
    # 2. vlmap_memft/trainer.py
    ###############################

    if not config.skip_vlmap:
        cmds = []
        for depth in DEPTHS:
            for vlmap_model in VLMAP_MODELS:
                for vlmap_seed in VLMAP_SEEDS:
                    cmd = 'python vlmap_memft/trainer.py' \
                        ' --model_type={vlmap_model}' \
                        ' --prefix={vlmap_prefix}' \
                        ' --max_train_iter=4810 --seed={vlmap_seed} --expand_depth={depth}' \
                        .format(vlmap_prefix=vlmap_prefix,
                                vlmap_model=vlmap_model,
                                vlmap_seed=vlmap_seed,
                                depth=depth)
                    if vlmap_model == '':
                        cmd += ' --enwiki_preprocessing=0'

                    cmds.append(cmd)

        parallel_run(cmds, config)
    
    #########################################
    # 3. symlink to experiments/important/*
    #########################################

    important_dir = "{}/{}".format(config.result_dir, TAG)
    makedirs(important_dir)

    important_sub_dirs = []
    dirs = list_dir("train_dir", prefix=VLMAP_BASE.format(
        vlmap_prefix="*",
        vlmap_model="*",
        depth="*",
        seed="*"))

    base_path = "{}/{}".format(config.result_dir, TAG)
    for directory in dirs:
        dir_time_str = directory.rsplit('_', 1)[1]

        if time_str <= dir_time_str:
            src_path = directory
            dst_path = os.path.join(base_path, directory.rsplit('/')[-1].rsplit('_', 1)[0])

            symlink(src_path, dst_path)
            important_sub_dirs.append(dst_path)
    
    #########################################
    # 4. vlmap_memft/export_word_weights.py
    #########################################

    cmds = []
    steps = {}

    for directory in important_sub_dirs:
        checkpoint_found = False
        for tmp_step in [4800, 4801]:
            check_path = os.path.join(directory, "model-{}.data-00000-of-00001".format(tmp_step))
            if os.path.exists(check_path):
                steps[directory] = tmp_step
                checkpoint_found = True
                break

        if not checkpoint_found:
            log.error("="*40)
            log.error(" {} not found".format(check_path))
            log.error("="*40)

        check_dir = os.path.join(directory, "word_weights_model-{}".format(steps[directory]))
        if os.path.exists(check_dir):
            continue
        checkpoint = os.path.join(directory, "model-{}".format(steps[directory]))

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
        for directory in important_sub_dirs:
            for vqa_seed in VQA_SEEDS:
                dp = find_arg(directory, "dp")
                vlmap_seed = find_arg(directory, "seed")

                base_cmd = "python vqa/trainer.py" \
                    " --vlmap_word_weight_dir {directory}/word_weights_model-{step}" \
                    " --seed {vqa_seed}" \
                    .format(directory=directory, dp=dp, step=steps[directory],
                            vlmap_seed=vlmap_seed, vqa_seed=vqa_seed)

                for model_type in MODEL_TYPES:
                    if model_type == 'standard_word2vec':
                        if 'vlmap_bf_or_wordset_withatt_sp' not in directory:
                            continue
                        cmd = base_cmd
                    elif model_type == 'vlmap_answer':
                        if 'vlmap_bf_or_wordset_withatt_sp' in directory:
                            continue
                        cmd = base_cmd + " --pretrained_param_path {directory}/model-{step}". \
                            format(directory=directory, step=steps[directory])
                    else:
                        raise Exception()

                    cmd += " --prefix dp{dp}_md{model_type}_sd{vlmap_seed}_vqasd{vqa_seed}". \
                        format(directory=directory, dp=dp, step=steps[directory],
                               vlmap_seed=vlmap_seed, vqa_seed=vqa_seed, model_type=model_type)

                    cmd += " --model_type={}".format(model_type)
                    cmds.append(cmd)

        parallel_run(cmds, config)

    #########################################
    # -2. vqa/eval_multiple_model.py
    #########################################

    cmd = "python vqa/eval_multiple_model.py --root_train_dir=train_dir"
    run(cmd, config)

    #########################################
    # -1. vqa/eval_collection.py
    #########################################

    cmd = "python vqa/eval_collection.py --root_train_dir=train_dir"
    run(cmd, config)
