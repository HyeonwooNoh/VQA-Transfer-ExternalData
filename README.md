# vlmap
Learning vision and language mapping as a visual knowledge base


## training conditional classifier

    vlmap/vlmap_mult_seed_run.sh

## Create weight_dir

    python vlmap_memft/export_word_weights.py --checkpoint experiments/important/0501_vlmap_ordered_iter_bf_or_wordset_seed_234_345_456/vlmap_bf_or_wordset_withatt_sp_d_memft_all_new_vocab50_obj3000_attr1000_maxlen10_ordered_iter_bs512_lr0.001_seed456_20180501-104510/model-4801

## training vqa

    vlmap/vqa_train_multseed.sh

or 

    vlmap/vqa_train_reproduce_test.sh


## Evaluation

    python vqa/eval_multiple_model.py --root_train_dir train_dir

이런식으로 여러개 training dir이 저장된 바로 윗 parent directory를 경로로 잡아서 eval_multiple_model 돌리면 각 run마다 있는 모든 checkpoint에 evaluation하고

    python vqa/eval_collection.py --root_train_dir train_dir

이걸 돌리면 evaluation 결과를 다 종합해서 "collect_eval_test_result.pkl" 이란걸 만들어주는데 [iteration, test result] plot 만드는데 쓸 데이타를 만드는거
이 다음에 ipython 돌리면 되

    http://147.47.209.134:10000/notebooks/plot_for_paper.ipynb
