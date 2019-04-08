# VQA-Transfer-ExternalData
This is not an official repository for the paper **Transfer Learning via Unsupervised Task Discovery for Visual Question Answering**.
Official repository for the paper is [https://github.com/HyeonwooNoh/vqa_task_discovery/](https://github.com/HyeonwooNoh/vqa_task_discovery/)


## All in 1

    python run.py 3 4 --time_str="20180508-145023" --skip_vlmap=1

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

Use direct parent of the target trainig directory as root_train_dir. Running eval_multiple_model script will evalute all the checkpionts in each run.

    python vqa/eval_collection.py --root_train_dir train_dir

After running eval_multiple_model script, this script will summarize all evaluation results and generate "collect_eval_test_result.pkl". The "collect_eval_test_result.pkl" contains raw data of [iteration, test results], which will be used for plotting. Please refer to the following ipython notebook for plotting examples.

    http://147.47.209.134:10000/notebooks/plot_for_paper.ipynb
