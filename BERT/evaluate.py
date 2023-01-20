import json

from seqeval.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score
from seqeval.scheme import IOB2, IOB1

from BERT.Model import NERModel
from util import xlarge
from util.list_utils import list_size, flatten_list
from util.text_utils import exclude_long_sentences


def evaluate(model: NERModel, test_sentences: list, test_labels: list, save_dir: str = None, print_report: bool = True):
    # Convert to BERT data model
    # test_x, test_y = bert_utils.dataset_to_bert_input(test_sentences, test_labels, model.tokenizer, model.vocabulary)

    test_sentences, test_labels = exclude_long_sentences(511, test_sentences, test_labels)

    # Predict outputs
    data_x = model.prepare_sentences(test_sentences)
    predicted_labels = model.predict(data_x, display_progress=True)
    predicted_labels = [[l if l != "[PAD]" else "O" for l in label] for label in predicted_labels]

    data_x = model.convert_ids_to_tokens(data_x)
    # Normalize to same tokenization as BERT
    # test_sentences, test_labels = model.normalize_tagged_dataset(test_sentences, test_labels)
    data_x, predicted_labels = model.align(test_sentences, data_x, predicted_labels)

    # Evaluate model
    if not (list_size(test_sentences) == list_size(data_x) == list_size(test_labels) == list_size(predicted_labels)):
        tmp_gl = []
        tmp_tl = []
        for gs, gl, ts, tl in zip(test_sentences, test_labels, data_x, predicted_labels):
            if len(gs) == len(gl) == len(ts) == len(tl):
                tmp_gl.append(gl)
                tmp_tl.append(tl)
                continue
            print("Sentence length mismatch")
            print(len(gs), len(gl), len(ts), len(tl))
            print("GS: ", gs)
            print("TS: ", ts)
        test_labels = tmp_gl
        predicted_labels = tmp_tl

    metrics = {
        'accuracy': accuracy_score(test_labels, predicted_labels),
        'precision': precision_score(test_labels, predicted_labels),
        'recall': recall_score(test_labels, predicted_labels),
        'f1': f1_score(test_labels, predicted_labels),
        # 'report': classification_report(test_labels, predicted_labels, scheme=IOB2),
        'strawberry': xlarge.score_from_iob(flatten_list(test_labels), flatten_list(predicted_labels), print_results=True),
    }

    if print_report:
        print('Accuracy: ' + str(metrics['accuracy']))
        print('Precision: ' + str(metrics['precision']))
        print('Recall: ' + str(metrics['recall']))
        print('F1 score: ' + str(metrics['f1']))
        # print(metrics['report'])
        print('Strawberry: ' + str(metrics['strawberry']))

    if save_dir is not None:
        with open(save_dir + '/test_metrics.txt', 'w') as f:
            json.dump(metrics, f, indent=4)

    # metrics.pop('report')
    return metrics
