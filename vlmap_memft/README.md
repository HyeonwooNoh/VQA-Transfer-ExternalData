Training with enwiki script:

    python vlmap_memft/trainer.py --model_type vlmap_wordset_only_withatt_sp
    python vlmap_memft/trainer.py --model_type vlmap_bf_or_wordset_enwiki_withatt_sp

Pretrained parameter export - example script:

    python vlmap_memft/export_word_weights.py --checkpoint experiments/important/0412_used_pretrained_vlmaps/vlmap_bf_or_wordset_withatt_sp_d_memft_all_new_vocab50_obj3000_attr1000_maxlen10_default_bs512_lr0.001_20180419-092348/model-4801

