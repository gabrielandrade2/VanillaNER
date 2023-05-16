#!/bin/bash
for noise in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
    for i in {0..10};
    do
        python3 main_conll_aida_yago.py --dataset resources/noise_1/noise_${noise}/AIDA-YAGO2-dataset_${i}.tsv \
          --model_path results/trained/noise_${noise}/model_${i} \
          --device cuda:0 \
          --max_epochs 10 \
          --batch_size 16 \
          --learning_rate 0.00001
    done
    wait
done
