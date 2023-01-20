import json
import os
import re

import math
import matplotlib.pyplot as plt
import mojimoji
import numpy as np
import torch
from seqeval.metrics import f1_score, accuracy_score
from tokenizations import tokenizations
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, BertJapaneseTokenizer, BertForTokenClassification, \
    AutoTokenizer, AutoModelForTokenClassification

from BERT import data_utils
from util import text_utils
from util.list_utils import list_size

CLS_TAG = '[CLS]'
PAD_TAG = '[PAD]'
UNK_TAG = '[UNK]'


class TrainingParameters:

    @classmethod
    def from_args(cls, args):
        parameters = TrainingParameters()
        parameters.__dict__.update(
            (k, args.__dict__[k]) for k in parameters.__dict__.keys() & args.__dict__.keys() if args.__dict__[k])
        return parameters

    @classmethod
    def add_parser_arguments(cls, parent_parser):
        parent_parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs')
        parent_parser.add_argument('--learning_rate', type=float, help='Learning rate')
        parent_parser.add_argument('--batch_size', type=int, help='Batch size')
        parent_parser.add_argument('--max_length', type=int, help='Maximum length of the input sequence')
        return parent_parser

    def __init__(self):
        # Load default parameters
        self.__dict__ = {
            'max_epochs': 10,
            'learning_rate': 1e-5,
            'batch_size': 16,
            'max_length': 512,
            'optimizer': optim.AdamW,
        }

    def set_max_epochs(self, max_epochs):
        self.__dict__['max_epochs'] = max_epochs
        return self

    def set_batch_size(self, batch_size):
        self.__dict__['batch_size'] = batch_size
        return self

    def set_learning_rate(self, learning_rate):
        self.__dict__['learning_rate'] = learning_rate
        return self

    def set_max_length(self, max_length):
        self.__dict__['max_length'] = max_length
        return self

    def set_optimizer(self, optimizer):
        self.__dict__['optimizer'] = optimizer
        return self


class NERModel:
    def __init__(self, pre_trained_model, tokenizer, vocabulary, device='cpu'):
        self.model = pre_trained_model
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.device = device
        self.max_size = None

    @classmethod
    def load_transformers_model(cls, pre_trained_model_name, model_dir, device='cpu', local_files_only=False):
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name, local_files_only=local_files_only)

        with open(model_dir + '/label_vocab.json', 'r') as f:
            label_vocab = json.load(f)

        model = AutoModelForTokenClassification.from_pretrained(pre_trained_model_name, num_labels=len(label_vocab),
                                                           local_files_only=local_files_only)
        model_path = model_dir + '/last_epoch.model'
        model.load_state_dict(torch.load(model_path, map_location=device))

        return cls(model, tokenizer, label_vocab, device)

    def set_max_size(self, max_size):
        self.max_size = max_size

    def train(self, x, y, parameters: TrainingParameters = None, val=None, outputdir=None):
        model = self.model
        device = self.device
        max_size = self.max_size
        self.training_metrics = {}

        if not max_size:
            max_size = max([len(i) for i in x])

        # Parse parameters
        if parameters is None:
            parameters = TrainingParameters()
        max_epoch = parameters.max_epochs
        batch_size = parameters.batch_size
        lr = parameters.learning_rate

        os.makedirs(outputdir, exist_ok=True)

        data = data_utils.Batch(x, y, batch_size=batch_size, max_size=max_size, sort=True)
        if val is not None:
            val_data = data_utils.Batch(val[0], val[1], batch_size=batch_size, max_size=max_size, sort=True)
            val_loss = []
            all_f1 = []
            all_acc = []
            lowest_loss = None
            lowest_loss_epoch = None
            highest_f1 = None
            highest_f1_epoch = None

        optimizer = parameters.optimizer(model.parameters(), lr=lr)
        total_step = int((len(data) // batch_size) * max_epoch)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_step * 0.1), total_step)

        losses = []
        model.to(device)
        with tqdm(range(max_epoch), desc="epoch", ncols=100, position=0, leave=True) as t:
            for epoch in t:
                model.train()
                all_loss = 0
                step = 0

                for sent, label, _ in tqdm(data, desc="batch", ncols=100,
                                           total=math.ceil(len(data) / data.batch_size),
                                           position=0, leave=True):
                    sent = torch.tensor(sent).to(device)
                    label = torch.tensor(label).to(device)
                    mask = [[float(i > 0) for i in ii] for ii in sent]
                    mask = torch.tensor(mask).to(device)

                    output = model(sent, attention_mask=mask, labels=label)
                    loss = output[0]
                    all_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                    step += 1

                losses.append(all_loss / step)

                if val is not None:
                    model.eval()
                    all_loss = 0
                    step = 0

                    gold = []
                    pred = []

                    for sent, label, _ in tqdm(val_data, desc="validation", ncols=100,
                                               total=math.ceil(len(val_data) / val_data.batch_size),
                                               position=0, leave=True):
                        sent = torch.tensor(sent).to(device)
                        label = torch.tensor(label).to(device)
                        mask = [[float(i > 0) for i in ii] for ii in sent]
                        mask = torch.tensor(mask).to(device)

                        output = model(sent, attention_mask=mask, labels=label)
                        loss = output[0]
                        all_loss += loss.item()

                        # Store gold and prediction to calculate metrics
                        gold.extend(label.detach().cpu().numpy()[:, 1:].tolist())
                        pred.extend(np.argmax(output[1].detach().cpu().numpy(), axis=2)[:, 1:].tolist())

                        step += 1

                    val_loss.append(all_loss / step)
                    if lowest_loss is None or lowest_loss > val_loss[-1]:
                        lowest_loss = val_loss[-1]
                        lowest_loss_epoch = epoch
                        torch.save(model.state_dict(), outputdir + '/lowest_loss.model')

                    # Calculate metrics
                    val_sentences = val_data.get_sentences()
                    gold = self.__remove_label_padding(val_sentences, gold)
                    pred = self.__remove_label_padding(val_sentences, pred)
                    gold = self.convert_prediction_to_labels(gold)
                    pred = self.convert_prediction_to_labels(pred)
                    all_f1.append(f1_score(gold, pred))
                    all_acc.append(accuracy_score(gold, pred))

                    if highest_f1 is None or highest_f1 < all_f1[-1]:
                        highest_f1 = all_f1[-1]
                        highest_f1_epoch = epoch
                        # torch.save(model.state_dict(), outputdir + '/highest_f1.model')

                    # output_path = outputdir + '/checkpoint{}.model'.format(len(val_loss) - 1)
                    # torch.save(model.state_dict(), output_path)

                t.set_postfix(loss=losses[-1], val_loss=val_loss[-1] if val is not None else 0,
                              val_f1=all_f1[-1] if val is not None else 0)

        # Save last epoch
        output_path = outputdir + '/last_epoch.model'
        torch.save(model.state_dict(), output_path)

        if val is not None:
            best_epoch = lowest_loss_epoch
            print('BEST EPOCH:' + str(best_epoch))
            model_path = outputdir + '/lowest_loss.model'
            model.load_state_dict(torch.load(model_path))

        x = range(max_epoch)
        plt.plot(x, losses, label='train loss')
        if val is not None:
            plt.plot(x, val_loss, label='val loss')
            plt.plot(x, all_f1, label='val f1')
        plt.xlabel('epoch')
        plt.title('best epoch: {}'.format(best_epoch))
        plt.legend()
        plt.savefig(os.path.join(outputdir, 'training.png'))
        plt.show()

        torch.save(model.state_dict(), outputdir + '/final.model')
        self.training_metrics['loss'] = losses[best_epoch]
        self.training_metrics['val_loss'] = val_loss[best_epoch]
        self.training_metrics['val_f1'] = all_f1[best_epoch]
        self.training_metrics['val_accuracy'] = all_acc[best_epoch]
        self.training_metrics['best_epoch'] = best_epoch
        self.training_metrics['lowest_loss_epoch'] = lowest_loss_epoch
        self.training_metrics['lowest_loss'] = lowest_loss
        self.training_metrics['highest_f1_epoch'] = highest_f1_epoch
        self.training_metrics['highest_f1'] = highest_f1
        with open(outputdir + '/training_metrics.txt', 'w') as f:
            json.dump(self.training_metrics, f, indent=4)

        self.model = model
        return model

    def predict(self, x, return_labels=True, display_progress=False):
        model = self.model
        device = self.device
        max_size = self.max_size
        all_loss = 0

        if not max_size:
            max_size = max([len(i) for i in x])

        data = data_utils.Batch(x, x, batch_size=8, sort=False, max_size=max_size)

        model.to(device)
        model.eval()

        res = []

        for sent, _, _ in tqdm(data, desc="prediction", ncols=100, total=int(len(data) / data.batch_size),
                               disable=not display_progress,
                               position=0, leave=True):
            sent = torch.tensor(sent).to(device)
            mask = [[float(i > 0) for i in ii] for ii in sent]
            mask = torch.tensor(mask).to(device)

            output = model(sent, attention_mask=mask)
            logits = output[0].detach().cpu().numpy()
            tags = np.argmax(logits, axis=2)[:, 1:].tolist()
            res.extend(tags)

        res = self.__remove_label_padding(x, res)

        if return_labels:
            return self.convert_prediction_to_labels(res)
        else:
            return res

    def prepare_sentences(self, sentences, split_sentences=False):
        """ Prepare sentences from model execution (tokenization + CLS tag addition).

        :param sentences: The list of sentences to be prepared.
        :param split_sentences: Should texts be split into sentences?
        :return: The list of prepared sentences.
        """
        if isinstance(sentences, str):
            temp = list()
            temp.append(sentences)
            sentences = temp

        if isinstance(sentences, list):
            if isinstance(sentences[0], list):
                sentences = [' '.join(sent) for sent in sentences]

        if split_sentences:
            single_item = len(sentences) == 1
            sentences = text_utils.split_sentences(sentences, single_item)

        # sentences = [' '.join(s) for s in self.__convert_to_zenkaku(sentences)]
        tokenized_sentences = [self.tokenizer.tokenize(t) for t in sentences]
        return [self.tokenizer.convert_tokens_to_ids([CLS_TAG] + t + ['[SEP]']) for t in tokenized_sentences]

    def normalize_tagged_dataset(self, sentences, tags):
        """ Use the tokenizer to apply the same normalization applied to full strings to a pre-tokenized dataset. It also
        adjusts the referring tags in case of removal/expansion of tokens.

        :param sentences: A list of list of character tokenized sentences.
        :param tags: A list of list of tags corresponding to the sentences.
        :return: Two lists, the processed sentences and the adjusted labels.
        """

        # processed_sentences = list()
        # processed_tags_sentences = list()
        #
        # for sentence, tag_sentence in zip(sentences, tags):
        #     processed_sentence = list()
        #     processed_tag_sentence = list()
        #     for character, tag_character in zip(sentence, tag_sentence):
        #         tokenized = self.tokenizer.tokenize(character)
        #         tokenized = re.findall(r"[\w]+|[^a-zA-Z0-9\s]", self.tokenizer.convert_tokens_to_string(tokenized))
        #             # mojimoji.han_to_zen(character) if character not in [CLS_TAG, PAD_TAG, UNK_TAG] else character)
        #         last_tag = str()
        #         for token in tokenized:
        #             if token == '' or token == ' ':
        #                 continue
        #             processed_sentence.append(token)
        #
        #             # In the case we are expanding a character that is tagged with a Beggining tag, we make the subsequent
        #             # ones as Intra.
        #             if last_tag.startswith('B') and last_tag == tag_character:
        #                 tag_character = tag_character.replace('B', 'I', 1)
        #             processed_tag_sentence.append(tag_character)
        #             last_tag = tag_character
        #
        #     processed_sentences.append(processed_sentence)
        #     processed_tags_sentences.append(processed_tag_sentence)
        # return processed_sentences, processed_tags_sentences
        processed_sentences = list()
        processed_tags_sentences = list()

        for sentence, tag_sentence in zip(sentences, tags):
            processed_sentence = list()
            processed_tag_sentence = list()
            for character, tag_character in zip(sentence, tag_sentence):
                tokenized = self.tokenizer.tokenize(
                    mojimoji.han_to_zen(character) if character not in [CLS_TAG, PAD_TAG, UNK_TAG] else character)
                last_tag = str()
                for token in tokenized:
                    if token == '' or token == ' ':
                        continue
                    processed_sentence.append(token)

                    # In the case we are expanding a character that is tagged with a Beggining tag, we make the subsequent
                    # ones as Intra.
                    if last_tag.startswith('B') and last_tag == tag_character:
                        tag_character = tag_character.replace('B', 'I', 1)
                    processed_tag_sentence.append(tag_character)

            processed_sentences.append(processed_sentence)
            processed_tags_sentences.append(processed_tag_sentence)
        return processed_sentences, processed_tags_sentences

    def convert_prediction_to_labels(self, prediction):
        id2label = {v: k for k, v in self.vocabulary.items()}
        return [[id2label[t] for t in tag] for tag in prediction]

    def __remove_label_padding(self, sentences, labels):
        new_labels = list()
        for sent, label in zip(sentences, labels):
            new_labels.append(label[:len(sent) - 2])

        # new_labels = [l if l != self.vocabulary[PAD_TAG] else self.vocabulary["O"] for l in new_labels]
        return new_labels

    def convert_ids_to_tokens(self, embeddings):
        temp = [self.tokenizer.convert_ids_to_tokens(t)[1:] for t in embeddings]
        temp = [[t for t in t_ if t != '[SEP]'] for t_ in temp]
        return self.__convert_to_zenkaku(temp)


    def align(self, sentences, predicted_sentences, predicted_labels):
        """ Align the predicted labels with the original sentences.

        :param sentences: The original sentences.
        :param predicted_sentences: The predicted sentences.
        :param predicted_labels: The predicted labels.
        :return: A list of tuples (predicted sentence, predicted labels).
        """
        mappings = [tokenizations.get_alignments(t, v)[0] for t, v in zip(sentences, predicted_sentences)]
        aligned_sentences = []
        aligned_labels = []
        i = 0
        # a = [['Dhaka', 'stocks', 'seen', 'steady', 'in', 'absence', 'of', 'big', 'players', '.']]
        # if sentences == a:
        #     print('stop')

        for s_,l_, m_ in zip(predicted_sentences, predicted_labels, mappings):
            temp_sentence = []
            temp_labels = []
            for m in m_:
                temp_token = ""
                if len(m) > 0:
                    for index in m:
                        temp_token += s_[index].replace("##", "")
                    temp_labels.append(l_[m[0]])
                    temp_sentence.append(temp_token)

            # for s, l, m in zip(s_, l_, m_):
            #     if len(m) > 0:
            #         if index != m[0]:
            #             if index is not None:
            #                 temp_sentence.append(temp_token)
            #                 temp_token = s.replace("##", "")
            #             else:
            #                 temp_token = temp_token + s.replace("##", "")
            #             temp_labels.append(l)
            #             index = m[0]
            #         else:
            #             temp_token = temp_token + s.replace("##", "")
            # if temp_token != "":
            #     temp_sentence.append(temp_token)
            if len(temp_sentence) != len(temp_labels) or len(temp_sentence) != len(sentences[i]):
                print('failed')
            aligned_sentences.append(temp_sentence)
            aligned_labels.append(temp_labels)
            i += 1

        return aligned_sentences, aligned_labels

    def __convert_to_zenkaku(self, tokens):
        return tokens
        # return [[mojimoji.han_to_zen(t) if t not in [CLS_TAG, PAD_TAG, UNK_TAG] else t for t in sent] for sent in
        #         tokens]

    def get_training_metrics(self):
        return self.training_metrics


    # def convert_ids_to_tokens(self, embeddings):
    #     temp = [self.tokenizer.convert_ids_to_tokens(t)[1:-1] for t in embeddings]
    #     temp = [re.findall(r"[\w]+|[^a-zA-Z0-9\s]", self.tokenizer.convert_tokens_to_string(t)) for t in temp]
    #     return self.__convert_to_zenkaku(temp)
    #
    #
    # def convert_ids_to_tokens_tags(self, embeddings, tags):
    #     temp = [self.tokenizer.convert_ids_to_tokens(t)[1:-1] for t in embeddings]
    #     temp2 = [re.findall(r"[\w]+|[^a-zA-Z0-9\s]", self.tokenizer.convert_tokens_to_string(t)) for t in temp]
    #
    #     mappings = [tokenizations.get_alignments(t, v)[1] for t,v in zip(temp, temp2)]
    #     i = 0
    #     out_tags = []
    #     for t_, m_ in zip(tags, mappings):
    #         index = None
    #         temp_tags = []
    #         for t,m in zip(t_, m_):
    #             if len(m) > 0:
    #                 if index != m[0]:
    #                     index = m[0]
    #                     temp_tags.append(t)
    #             else:
    #                 temp2
    #         if len(temp_tags) != len(temp2[i]):
    #             print('failed')
    #         out_tags.append(temp_tags)
    #         i += 1
    #
    #     assert list_size(temp2) == list_size(out_tags)
    #     return self.__convert_to_zenkaku(temp2), tags