import os
from argparse import ArgumentParser
from random import random

import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

def parse_conll_ner_data(input_file: str, encoding: str = "utf-8"):
    words: List[str] = []
    labels: List[str] = []
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
                        yield words, labels, p1, p2, sentence_boundaries
                        words = []
                        labels = []
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
                        p1.append('\t'.join(parts[2:]))
                    else:
                        labels.append("O")
                        p1.append('')


        if words:
            yield words, labels, p1, p2, sentence_boundaries
    except UnicodeDecodeError as e:
        raise Exception("The specified encoding seems wrong. Try either ISO-8859-1 or utf-8.") from e


def decision(probability):
    return random() < probability


if __name__ == '__main__':

    parser = ArgumentParser()
    # parser.add_argument("--noise_ratio", type=float, required=True)
    # parser.add_argument("--noise_amount", type=float, required=True)
    args, _ = parser.parse_known_args()

    # noise_ratio = args.noise_ratio
    # noise_amount = args.noise_amount

    for noise_amount in [2, 3]:
        for noise_ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            for file in ["/Users/gabriel-he/PycharmProjects/SimpleNER/resources/AIDA-YAGO2-dataset.tsv"]:
                for i in tqdm(range(3)):
                    iter = parse_conll_ner_data(file)
                    os.makedirs('resources/noise_{}/noise_{}'.format(noise_amount, noise_ratio), exist_ok=True)
                    filename = file.split('/')[-1].split('.')[0]
                    extension = file.split('/')[-1].split('.')[-1]
                    fout = open('resources/noise_{}/noise_{}/{}_{}.{}'.format(noise_amount, noise_ratio, filename, i, extension), 'w',
                                encoding='utf8')

                    for doc in iter:
                        if len(doc) <= 1:
                            fout.write(doc[0])
                            continue
                        labels = doc[1]
                        if ('testb' not in doc[3]):
                            for j in range(len(labels)):
                                if labels[j].startswith('B'):
                                    words = []
                                    if decision(noise_ratio):
                                        # 1 - backward_noise, 2 - forward_noise, 3 - both
                                        noise_type = np.random.choice([1, 2, 3])
                                        if noise_type & 1:
                                            try:
                                                max = 0
                                                edit = False
                                                for k in range(1, noise_amount + 1):
                                                    if j - k < 0 or (j-k+1 in doc[4]):
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
                                                    doc[2][j - k] = doc[2][j]
                                                    edit = True
                                                if edit:
                                                    labels[j] = 'I' + labels[j][1:]
                                            except IndexError:
                                                pass
                                        if noise_type & 2:
                                            try:
                                                while (labels[j + 1].startswith('I') and (j+1 not in doc[4])):
                                                    j += 1
                                                for k in range(1, noise_amount + 1):
                                                    # Commented to allow overlap
                                                    # if labels[j + k].startswith('O'):
                                                    #     labels[j + k] = 'I' + labels[j][1:]
                                                    #     doc[2][j + k] = doc[2][j]
                                                    # else:
                                                    #     break
                                                    labels[j + k] = 'I' + labels[j][1:]
                                                    doc[2][j + k] = doc[2][j]
                                            except IndexError:
                                                pass

                        fout.write(doc[3])
                        for j in range(len(doc[0])):
                            if j in doc[4]:
                                fout.write('\n')
                            if labels[j].startswith('O'):
                                fout.write(doc[0][j] + '\n')
                            else:
                                fout.write('{}\t{}\t{}\n'.format(doc[0][j], labels[j], doc[2][j]))
                        fout.write('\n')

                    fout.flush()
                    fout.close()
