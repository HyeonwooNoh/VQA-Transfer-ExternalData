dp='False'
prefix='expand_depth'
expdir=experiments/important/0504_vqa_bf_or_ws_123_from_vlamp_234_345_456_depth${dp}
modelprefix=vlmap_bf_or_wordset_withatt_sp_d_memft_all_new_vocab50_obj3000_attr1000_maxlen10_${prefix}_bs512_lr0.001

for vlseed in 234 345 456; do
    for seed in 123 234 345; do
        CUDA_VISIBLE_DEVICES=1 python vqa/trainer.py --pretrained_param_path ${expdir}/${modelprefix}_dp${dp}_seed${vlseed}/model-4800 --vlmap_word_weight_dir ${expdir}/${modelprefix}_dp${dp}_seed${vlseed}/word_weights_model-4800 --prefix bf_or_wordset_withatt_sp_ordered_iter_sd${vlseed} --seed $seed
    done
done

#expdir=experiments/important/0501_vlmap_ordered_iter_bf_or_wordset_seed_234_345_456
#modelprefix=vlmap_bf_or_wordset_withatt_sp_d_memft_all_new_vocab50_obj3000_attr1000_maxlen10_ordered_iter_bs512_lr0.001
#
#for vlseed in 234 345 456
#do
#    for seed in 123 234 345
#    doseed 
#        python vqa/trainer.py --pretrained_param_path ${expdir}/${modelprefix}_seed${vlseed}/model-4800 --vlmap_word_weight_dir ${expdir}/${modelprefix}_seed${vlseed}/word_weights_model-4800 --prefix bf_or_wordset_withatt_sp_ordered_iter_sd${vlseed} --seed $seed
#    done
#done
