import argparse
import json
import os

import torch
from seqeval.metrics import accuracy_score, recall_score, f1_score, precision_score
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from BERT import bert_utils
from BERT.Model import NERModel
from BERT.Model import TrainingParameters
from BERT.evaluate import evaluate
from BERT.train import finetune_from_sentences_tags_list
from util import strawberry
from util.list_utils import flatten_list


class Dataset:

    def __init__(self):
        self.documents = []

    def append_document(self, document):
        self.documents.append(document)

    def get_sentences_labels(self):
        sentences = []
        labels = []
        for document in self.documents:
            sentences.extend(document.sentences)
            labels.extend(document.labels)
        return sentences, labels


class Document:

    def __init__(self, sentences, labels, others, id=None, title=None):
        self.id = id
        self.title = title
        self.sentences = sentences
        self.labels = labels
        self.other = others

def parse_aida_yago(file_path: str):
    fIn = open(file_path, 'r')

    train = Dataset()
    testa = Dataset()
    testb = Dataset()
    doc_sentences = []
    doc_labels = []
    doc_others = []
    sentences = []
    labels = []
    others = []

    for line in fIn:
        if line.startswith('-DOCSTART-'):
            lastNER = 'O'
            if sentences or doc_sentences:
                if sentences:
                    doc_sentences.append(sentences)
                    doc_labels.append(labels)
                    doc_others.append(others)

                doc = Document(doc_sentences, doc_labels, doc_others, id=id, title=title)

                if "testa" in id:
                    testa.append_document(doc)
                elif "testb" in id:
                    testb.append_document(doc)
                else:
                    train.append_document(doc)

                doc_sentences = []
                doc_labels = []
                doc_others = []
                sentences = []
                labels = []
                others = []
            doc_metadata = line[line.find("(") + 1:line.find(")")].split(' ')
            id = doc_metadata[0]
            title = doc_metadata[1]
            continue

        if len(line.strip()) == 0:
            lastNER = 'O'
            if sentences:
                doc_sentences.append(sentences)
                doc_labels.append(labels)
                doc_others.append(others)
                sentences = []
                labels = []
                others = []
            continue

        splits = line.strip().split('\t')

        word = splits[0]
        ner = splits[1] if len(splits) > 1 else 'O'

        if ner[0] == 'I':
            if ner[1:] != lastNER[1:]:
                ner = 'B' + ner[1:]

        sentences.append(word)
        labels.append(ner)
        others.append(splits[2:] if len(splits) > 2 else [])

        lastNER = ner

    if sentences or doc_sentences:
        if sentences:
            doc_sentences.append(sentences)
            doc_labels.append(labels)
            doc_others.append(others)

        doc = Document(doc_sentences, doc_labels, doc_others, id=id, title=title)

        if "testa" in id:
            testa.append_document(doc)
        elif "testb" in id:
            testb.append_document(doc)
        else:
            train.append_document(doc)

    return train, testa, testb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Model')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--device', type=str, help='Device', default='cuda')
    TrainingParameters.add_parser_arguments(parser)
    args = parser.parse_args()

    import time

    start_time = time.time()

    model = "bert-base-cased"
    model_path = args.model_path

    # documents_train, documents_testa, documents_testb = parse_aida_yago('resources/AIDA-YAGO2-dataset.tsv')
    documents_train, documents_testa, documents_testb = parse_aida_yago(args.dataset)

    train, labels_train = documents_train.get_sentences_labels()
    testa, labels_testa = documents_testa.get_sentences_labels()
    testb, labels_testb = documents_testb.get_sentences_labels()

    print(max([len(x) for x in testa]))
    print(max([len(x) for x in testb]))
    print(max([len(x) for x in train]))

    parameters = TrainingParameters.from_args(args)

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Create vocabulary
    # label_vocab = dict(pre_trained_model.config.label2id)
    label_vocab = bert_utils.create_label_vocab(labels_train + labels_testa + labels_testb)
    pre_trained_model = AutoModelForTokenClassification.from_pretrained(model, num_labels=len(label_vocab))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

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
    with open(model_path+ '/output/testbout.txt', 'w') as outfile:
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
        'strawberry': strawberry.score_from_iob(test_labels_s, predicted_labels_s, print_results=True),
    }

    print('Accuracy: ' + str(metrics['accuracy']))
    print('Precision: ' + str(metrics['precision']))
    print('Recall: ' + str(metrics['recall']))
    print('F1 score: ' + str(metrics['f1']))

    output_dir = model_path + '/output'
    with open(output_dir + '/eval_results.txt', 'w') as outfile:
        json.dump(metrics, outfile, indent=4)

    print("--- %s seconds ---" % (time.time() - start_time))



