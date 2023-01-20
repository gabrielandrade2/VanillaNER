import glob
import os
import shutil
from random import random
from typing import List

import numpy as np
from tqdm import tqdm

def __parse_conll_ner_data(input_file: str, encoding: str = "utf-8"):
    words: List[str] = []
    labels: List[str] = []
    text: List[str] = []
    p1: List[str] = []
    p2: List[str] = []

    sentence_boundaries: List[int] = [0]

    try:
        with open(input_file, "r", encoding=encoding) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    if words:
                        assert sentence_boundaries[0] == 0
                        assert sentence_boundaries[-1] == len(words)
                        yield words, labels, text, p1, p2, sentence_boundaries
                        words = []
                        labels = []
                        text = []
                        p1 = []
                        p2 = []
                        sentence_boundaries = [0]
                    p2 = line
                    continue

                if not line:
                    if len(words) != sentence_boundaries[-1]:
                        sentence_boundaries.append(len(words))
                else:
                    parts = line.split("\t")
                    words.append(parts[0])
                    if len(parts) > 1:
                        labels.append(parts[1])
                        text.append(parts[2])
                        p1.append('\t'.join(parts[3:]))
                    else:
                        labels.append("O")
                        text.append("")
                        p1.append('')


        if words:
            yield words, labels, text, p1, p2, sentence_boundaries
    except UnicodeDecodeError as e:
        raise Exception("The specified encoding seems wrong. Try either ISO-8859-1 or utf-8.") from e

def __decision(probability):
    '''
    Random decision used to select if an annotation should or not be expanded.

    :param probability: The probability to be selected.
    :return: True or False.
    '''
    return random() < probability


def __generate_noise(dataset_files, word_amounts):
    folders = []

    if isinstance(dataset_files, str):
        dataset_files = [dataset_files]

    if isinstance(word_amounts, int):
        word_amounts = [word_amounts]

    dataset_instances = 10

    for word_amount in word_amounts:
        for noise_level in [p/10 for p in range(1, 11)]:
            folder = 'resources/noise_{}/noise_{}'.format(word_amount, noise_level)
            os.makedirs(folder, exist_ok=True)
            folders.append(folder)
            for file in dataset_files:
                for i in tqdm(range(dataset_instances), desc='Generating dataset instances'):
                    iter = __parse_conll_ner_data(file)
                    filename = file.split('/')[-1].split('.')[0]
                    extension = file.split('/')[-1].split('.')[-1]
                    fout = open(folder + '/{}_{}.{}'.format(filename, i, extension), 'w',
                                encoding='utf8')

                    for doc in iter:
                        if len(doc) <= 1:
                            fout.write(doc[0])
                            continue
                        labels = doc[1]

                        # Modify all documents except for the test set
                        if ('testb' not in doc[4]):
                            for j in range(len(labels)):
                                if labels[j].startswith('B'):
                                    words = []
                                    if __decision(noise_level):
                                        # 1 - backward_noise, 2 - forward_noise, 3 - both
                                        noise_type = np.random.choice([1, 2, 3])
                                        if noise_type & 1:
                                            try:
                                                max = 0
                                                edit = False
                                                for k in range(1, word_amount + 1):
                                                    if j - k < 0 or (j-k+1 in doc[5]):
                                                        break
                                                    max = k
                                                    # Commented to allow overlap
                                                    # if labels[j - k].startswith('O'):
                                                    #     max = k
                                                    # else:
                                                    #     break

                                                for k in range(1, max + 1):
                                                    if k == max:
                                                        labels[j - k] = 'B' + labels[j][1:]
                                                    else:
                                                        labels[j - k] = 'I' + labels[j][1:]
                                                    doc[3][j - k] = doc[3][j]
                                                    edit = True
                                                if edit:
                                                    labels[j] = 'I' + labels[j][1:]
                                            except IndexError:
                                                pass
                                        if noise_type & 2:
                                            try:
                                                while (labels[j + 1].startswith('I') and (j+1 not in doc[5])):
                                                    j += 1
                                                for k in range(1, word_amount + 1):
                                                    # Commented to allow overlap
                                                    # if labels[j + k].startswith('O'):
                                                    #     labels[j + k] = 'I' + labels[j][1:]
                                                    #     doc[2][j + k] = doc[2][j]
                                                    # else:
                                                    #     break
                                                    labels[j + k] = 'I' + labels[j][1:]
                                                    doc[3][j + k] = doc[3][j]
                                            except IndexError:
                                                pass

                        fout.write(doc[4])
                        for j in range(len(doc[0])):
                            if j in doc[5]:
                                fout.write('\n')
                            if labels[j].startswith('O'):
                                fout.write(doc[0][j] + '\n')
                            else:
                                fout.write('{}\t{}\t{}\n'.format(doc[0][j], labels[j], doc[3][j]))
                        fout.write('\n')

                    fout.flush()
                    fout.close()

    return folders
def __fix_mention_text(noisy_folders):
    '''
    Fix the mention text in the noisy datasets, to reflect the new span.
    '''

    for folder in noisy_folders:
        folder_fix = folder.replace('resources/noise_', 'resources/noise_fix_')
        for file in tqdm(glob.glob(folder + '*/*'), desc='Fixing span'):
            iter = __parse_conll_ner_data(file)
            filename = str(file).replace(folder, folder_fix)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fout = open(filename, 'w')

            for doc in iter:
                words = []
                labels = doc[1]
                for j in range(len(labels)):
                    if labels[j].startswith('B'):
                        text = [doc[0][j]]
                        for k in range(j + 1, len(labels)):
                            if labels[k].startswith('I'):
                                text.append(doc[0][k])
                            else:
                                break
                        words.append(' '.join(text))


                i = 0
                for j in range(len(labels)):
                    if labels[j].startswith('B'):
                        doc[2][j] = words[i]
                        i += 1
                    if labels[j].startswith('I'):
                        doc[2][j] = words[i - 1]

                fout.write(doc[4])
                for j in range(len(doc[0])):
                    if j in doc[5]:
                        fout.write('\n')
                    if labels[j].startswith('O'):
                        fout.write(doc[0][j] + '\n')
                    else:
                        fout.write('{}\t{}\t{}\t{}\n'.format(doc[0][j], labels[j], doc[2][j], doc[3][j]))
                fout.write('\n')

            fout.flush()
            fout.close()
        shutil.rmtree(folder)
        shutil.move(folder_fix, folder)

def generate_noisy_datasets(dataset_files="resources/AIDA-YAGO2-dataset.tsv", word_amounts=[1]):
    '''
    Function used to generate boundary-expanded variants a dataset file in conll2003 format.

    :param dataset_files: A string or list of string with the path to the dataset files.
    :param word_amounts: An integer or list of integer specifying the amount of tokens to expand to either side of the
    annotation boundaries.
    '''
    folders = __generate_noise(dataset_files, word_amounts)
    __fix_mention_text(folders)
