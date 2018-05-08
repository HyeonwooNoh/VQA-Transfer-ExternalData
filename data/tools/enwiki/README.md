# enwiki preprocessing

preprocessing code in this directory is constructed based on the instruction of https://github.com/jind11/word2vec-on-wikipedia

    cd /home/hyeonwoonoh/vlmap/data/preprocessed/enwiki
    python data/tools/enwiki/1_merge_and_count.py
    python data/tools/enwiki/2_word2contexts.py --preprocessing=0
    python data/tools/enwiki/3_make_wordset.py --preprocessing=0
    python data/tools/enwiki/2_word2contexts.py --preprocessing=1
    python data/tools/enwiki/3_make_wordset.py --preprocessing=1

