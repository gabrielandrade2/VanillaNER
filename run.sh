#!/bin/bash
for noise in 0.05 0.1 0.2 0.3 0.6 1.0;
do
    for i in {0..2};
    do
        python3 main_conll_aida_yago.py --dataset /home/is/gabriel-he/pycharm-upload/SimpleNER/resources/noise_fix/noise_${noise}/AIDA-YAGO2-dataset_${i}.tsv \
          --model_path /home/is/gabriel-he/pycharm-upload/SimpleNER/results/noise_fix_1_lr_30/noise_${noise}/model_${i} \
          --device cuda:0 \
          --max_epochs 10 \
          --batch_size 16 \
          --learning_rate 0.0003 &
    done
    wait
done
