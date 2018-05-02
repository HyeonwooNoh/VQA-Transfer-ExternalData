import os

from util import log


dirname, filename = os.path.split(os.path.abspath(__file__))
root_dir = "/".join(dirname.split("/")[:-3])


def symlink(config, filename):
    src_path = os.path.join(root_dir, config.reference_vqa_dir, filename)
    dst_path = os.path.join(root_dir, config.caption_split_dir, filename)

    if os.path.exists(dst_path):
        log.info("{} already exists".format(dst_path))
    else:
        log.info("Sym link: {}->{}".format(src_path, dst_path))
        os.symlink(src_path, dst_path)
