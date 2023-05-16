#!/bin/bash
for noise in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
    for i in {0..10};
    do
        python3 predict_conll_aida_yago.py --dataset resources/AIDA-YAGO2-dataset.tsv \
          --model_path results/trained/noise_${noise}/model_${i} \
          --device cuda:0
    done
    wait
done
