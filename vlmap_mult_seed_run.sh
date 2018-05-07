python data/tools/visualgenome/find_word_group.py --expand_depth=True
CUDA_VISIBLE_DEVICES=0 python vlmap_memft/trainer.py --model_type vlmap_bf_or_wordset_withatt_sp --prefix expand_depth --max_train_iter 4810 --seed 234 --expand_depth True
CUDA_VISIBLE_DEVICES=0 python vlmap_memft/trainer.py --model_type vlmap_bf_or_wordset_withatt_sp --prefix expand_depth --max_train_iter 4810 --seed 345 --expand_depth True
CUDA_VISIBLE_DEVICES=0 python vlmap_memft/trainer.py --model_type vlmap_bf_or_wordset_withatt_sp --prefix expand_depth --max_train_iter 4810 --seed 456 --expand_depth True

python data/tools/visualgenome/find_word_group.py --expand_depth=False
#CUDA_VISIBLE_DEVICES=1 python vlmap_memft/trainer.py --model_type vlmap_bf_or_wordset_withatt_sp --prefix expand_depth --max_train_iter 4810 --seed 234 --expand_depth False
#CUDA_VISIBLE_DEVICES=1 python vlmap_memft/trainer.py --model_type vlmap_bf_or_wordset_withatt_sp --prefix expand_depth --max_train_iter 4810 --seed 345 --expand_depth False
#CUDA_VISIBLE_DEVICES=1 python vlmap_memft/trainer.py --model_type vlmap_bf_or_wordset_withatt_sp --prefix expand_depth --max_train_iter 4810 --seed 456 --expand_depth False
