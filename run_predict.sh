#!/bin/bash
for noise in 0.05 0.1 0.2 0.3 0.6 1.0;
do
    for i in {0..2};
    do
        python3 predict_conll_aida_yago.py --dataset /home/is/gabriel-he/pycharm-upload/SimpleNER/resources/AIDA-YAGO2-dataset.tsv \
          --model_path /home/is/gabriel-he/pycharm-upload/SimpleNER/results/noise_fix_1_train_fix/noise_${noise}/model_${i} \
          --device cuda:1 &
    done
    wait
done
