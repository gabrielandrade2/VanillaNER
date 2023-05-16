import argparse
import json
import os

import torch
from seqeval.metrics import accuracy_score, recall_score, f1_score, precision_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from BERT import bert_utils
from BERT.Model import NERModel
from BERT.Model import TrainingParameters
from BERT.evaluate import evaluate
from BERT.train import finetune_from_sentences_tags_list
from util.Dataset import parse_aida_yago
from util.list_utils import flatten_list
from util.EL_prediction_file_util import add_gold_entity_to_NER_iob_output

'''
Script used for training a BERT model on the AIDA-CoNLL-YAGO dataset and evaluating it on the testb set.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Model', default='results/trained')
    parser.add_argument('--dataset', type=str, help='Dataset', default='resources/AIDA-YAGO2-dataset.tsv')
    parser.add_argument('--device', type=str, help='Device', default='cuda')
    TrainingParameters.add_parser_arguments(parser)
    args = parser.parse_args()

    import time

    start_time = time.time()

    model = "bert-base-cased"
    model_path = args.model_path

    documents_train, documents_testa, documents_testb = parse_aida_yago(args.dataset)

    train, labels_train = documents_train.get_sentences_labels()
    testa, labels_testa = documents_testa.get_sentences_labels()
    testb, labels_testb = documents_testb.get_sentences_labels()

    print(max([len(x) for x in testa]))
    print(max([len(x) for x in testb]))
    print(max([len(x) for x in train]))

    parameters = TrainingParameters.from_args(args)
    parameters.set_max_epochs(2)

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Create vocabulary
    label_vocab = bert_utils.create_label_vocab(labels_train + labels_testa + labels_testb)
    pre_trained_model = AutoModelForTokenClassification.from_pretrained(model, num_labels=len(label_vocab))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(device)

    print("Training...")
    model = NERModel(pre_trained_model, tokenizer, label_vocab, device=device)
    model.set_max_size(512)
    model = finetune_from_sentences_tags_list(train, labels_train, testa, labels_testa, model, model_path, parameters=parameters)

    print("Testing...")
    evaluate(model, testa, labels_testa, save_dir=model_path)

    print("Predicting...")
    os.makedirs(model_path + '/output', exist_ok=True)
    test_sentences = []
    test_labels = []
    predicted_labels = []
    with open(model_path+ '/output/output.iob', 'w') as outfile:
        for test_document in tqdm(documents_testb.documents, desc="Output prediction"):
            outfile.write('-DOCSTART-\t({} {})\n'.format(test_document.id, test_document.title))
            for test_sent, test_label in zip(test_document.sentences, test_document.labels):
                test_labels.append(test_label)
                test_sentences.append(test_sent)
                sentences_embeddings = model.prepare_sentences([test_sent], False)
                labels = model.predict(sentences_embeddings)
                labels = [[l if l != "[PAD]" else "O" for l in label] for label in labels]
                sentences = model.convert_ids_to_tokens(sentences_embeddings)
                sentences, labels = model.align([test_sent], sentences, labels)
                for l in labels:
                    predicted_labels.append(l)
                for sentence, label in zip(sentences, labels):
                    for word, tag in zip(sentence, label):
                        outfile.write(word.strip() + '\t' + tag + '\n')

    test_labels_s = [label + "-A" if label != 'O' else label for label in flatten_list(test_labels)]
    predicted_labels_s = [label + "-A" if label != 'O' else label for label in flatten_list(predicted_labels)]

    metrics = {
        'accuracy': accuracy_score(test_labels, predicted_labels),
        'precision': precision_score(test_labels, predicted_labels),
        'recall': recall_score(test_labels, predicted_labels),
        'f1': f1_score(test_labels, predicted_labels),
    }

    print('Accuracy: ' + str(metrics['accuracy']))
    print('Precision: ' + str(metrics['precision']))
    print('Recall: ' + str(metrics['recall']))
    print('F1 score: ' + str(metrics['f1']))

    output_dir = model_path + '/output'
    with open(output_dir + '/eval_results.txt', 'w') as outfile:
        json.dump(metrics, outfile, indent=4)

    add_gold_entity_to_NER_iob_output(output_dir + '/output.iob', 'resources/AIDA-YAGO2-dataset.tsv')

    print("--- %s seconds ---" % (time.time() - start_time))



