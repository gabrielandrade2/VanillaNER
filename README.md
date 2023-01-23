# VanillaNER

VanillaNER is a simple and low-effort Named Entity Recognition (NER) system, used for evaluation of boundary-relaxed named entity annotation in the paper "[Comparative evaluation of boundary-relaxed annotation for Entity Linking Performance](link)".

It is based on the `bert-base-cased` model provided by [HuggingFace](https://huggingface.co/bert-base-cased) library and compatible with Python 3.6+. 

The model was evaluated on the [AIDA-CoNLL-YAGO](https://resources.mpi-inf.mpg.de/yago-naga/aida/downloads.html) dataset.

In this repository we provide the scripts used for model finetunning and prediction, as well as the generation of the noisy dataset variants. The

## Requirements

Requirements are listed in `requirements.txt` and can be installed using `pip install -r requirements.txt`.

The code was originally run under `Python 3.9`, however it should work with Python 3.6+

## Usage

### Noise generation

Noisy dataset variants are produced by running `main_generate_noisy_dataset.py`.

This is script is hardcoded to look for the original dataset in `data/AIDA-YAGO2-dataset.tsv` and to save the generated variants in `resources/noise_1`.

This script will generate 10 dataset variants for noise levels from 10% to 100% in steps of 10%.

No parameters are provided for modifying the behavior of this script at this time, however it is possible to modify the script to change the number of variants generated, the noise levels, or the output directory by modifying the code, as all of these inputs are variable dependent.

### Finetunning

The script for model training is `main_conll_aida_yago.py`. 

This script is responsible for parsing the input dataset and use the predefined splits (train, testa and testb) for training, validation and testing the model.
For running some parameters are necessary, as described below:

- `--dataset`: Path for the dataset file where the dataset is located. Default: `resources/AIDA-YAGO2-dataset_0.tsv`
- `--model_path`: The output folder to store the produced model. Default: `results/test`
- `--device`: The device to use for training (cuda or cpu). Default: `cuda`

Example:

    python main_conll_aida_yago.py --dataset resources/AIDA-YAGO2-dataset_0.tsv --model_path results/test --device cuda

The produced model is stored as a common pytorch model and can be load used either Pytorch of HuggingFace libraries.

### Prediction
 
While `main_conll_aida_yago.py` script also performs prediction, a convenience script is provided to just run prediction without training the model, by running `predict_conll_aida_yago.py`.

The prediction script takes the same parameters as the finetunning script. Example:

    python predict_conll_aida_yago.py --dataset resources/AIDA-YAGO2-dataset_0.tsv --model_path results/test --device cuda

Output predictions (for both scripts) are saved in a new `output` folder under the `model_path` directory. 
- `output.iob` contains the predictions for the test set in IOB format.
- `eval_results.txt` contains the evaluation metrics calculated for the results (accuracy, precision, recall and F1).