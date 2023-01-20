import json
import os

import torch
from seqeval.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_score
from seqeval.scheme import IOB2
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import pipeline

from BERT import bert_utils
from BERT.Model import TrainingParameters
from BERT.evaluate import evaluate
from BERT.train import train_from_sentences_tags_list, finetune_from_sentences_tags_list
from util import xlarge
from util.list_utils import flatten_list


class Document:

    def __init__(self, sentences, labels, id=None, title=None):
        self.id = id
        self.title = title
        self.sentences = sentences
        self.labels = labels

def parse_conll2003(file_path: str):
    fIn = open(file_path, 'r')

    data = ([], [], [])
    doc_sentences = []
    doc_labels = []
    sentences = []
    labels = []

    for line in fIn:
        if line.startswith('-DOCSTART-'):
            lastNER = 'O'
            if sentences:
                data[0].append(sentences)
                data[1].append(labels)
                doc_sentences.append(sentences)
                doc_labels.append(labels)
                data[2].append(Document(doc_sentences, doc_labels))
                doc_sentences = []
                doc_labels = []
                sentences = []
                labels = []
            continue

        if len(line.strip()) == 0:
            lastNER = 'O'
            if sentences:
                data[0].append(sentences)
                data[1].append(labels)
                doc_sentences.append(sentences)
                doc_labels.append(labels)
                sentences = []
                labels = []
            continue

        splits = line.strip().split()

        word = splits[0]
        ner = splits[3]

        if ner[0] == 'I':
            if ner[1:] != lastNER[1:]:
                ner = 'B' + ner[1:]

        sentences.append(word)
        labels.append(ner)

        lastNER = ner

    # if sentences:
    #     data[0].append(sentences)
    #     data[1].append(labels)
    return data

if __name__ == '__main__':
    model = "bert-base-cased"
    model_path = 'results/finetuned_bert_base_casedtest'

    testa, labels_testa, documents_testa = parse_conll2003('resources/eng.testa')
    testb, labels_testb, documents_testb = parse_conll2003('resources/eng.testb')
    train, labels_train, documents_train = parse_conll2003('resources/eng.train')

    print(max([len(x) for x in testa]))
    print(max([len(x) for x in testb]))
    print(max([len(x) for x in train]))

    parameters = TrainingParameters()
    parameters.set_max_epochs(10)
    # parameters.set_batch_size(32)
    # parameters.set_learning_rate(5e-5)
    from BERT.Model import NERModel

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Create vocabulary
    # label_vocab = dict(pre_trained_model.config.label2id)
    label_vocab = bert_utils.create_label_vocab(labels_train + labels_testa + labels_testb)
    pre_trained_model = AutoModelForTokenClassification.from_pretrained(model, num_labels=len(label_vocab))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Training...")
    model = NERModel(pre_trained_model, tokenizer, label_vocab, device=device)
    model.set_max_size(512)
    model = finetune_from_sentences_tags_list(train, labels_train, testa, labels_testa, model, model_path, parameters=parameters)

    print("Testing...")
    evaluate(model, testa, labels_testa)

    print("Predicting...")
    os.makedirs(model_path + '/output', exist_ok=True)
    test_labels = []
    predicted_labels = []
    with open(model_path+ '/output/testbout.txt', 'w') as outfile:
        for test_document in documents_testb:
            for test_sent, test_label in zip(test_document.sentences, test_document.labels):
                test_labels.append(test_label)
                sentences_embeddings = model.prepare_sentences([test_sent], False)
                labels, _ = model.predict(sentences_embeddings)
                sentences = model.convert_ids_to_tokens(sentences_embeddings)
                labels = [[l if l != "[PAD]" else "O" for l in label] for label in labels]
                predicted_labels.append(labels)
            outfile.write('-DOCSTART- -X- -X- O\n\n')
            for sentence, label in zip(sentences, labels):
                for word, tag in zip(sentence, label):
                    outfile.write(word + ' ' + tag + ' \n')
                outfile.write('\n')


    with open(model_path+ '/output/eval_results.txt', 'w') as outfile:
        metrics = {
            'accuracy': accuracy_score(test_labels, predicted_labels),
            'precision': precision_score(test_labels, predicted_labels),
            'recall': recall_score(test_labels, predicted_labels),
            'f1': f1_score(test_labels, predicted_labels),
            'report': classification_report(test_labels, predicted_labels, scheme=IOB2),
            'strawberry': xlarge.score_from_iob(flatten_list(test_labels), flatten_list(predicted_labels),
                                                print_results=True)
        }
        print('Accuracy: ' + str(metrics['accuracy']))
        print('Precision: ' + str(metrics['precision']))
        print('Recall: ' + str(metrics['recall']))
        print('F1 score: ' + str(metrics['f1']))
        print(metrics['report'])
        print('Strawberry: ' + str(metrics['strawberry']))

    # example = "My name is Wolfgang and I live in Berlin"
    # sentences_embeddings = model.prepare_sentences([example], False)
    # labels = model.predict(sentences_embeddings)
    # sentences = model.convert_ids_to_tokens(sentences_embeddings)
    # labels = [[l if l != "[PAD]" else "O" for l in label] for label in labels]
    # print(sentences)
    # print(labels)
