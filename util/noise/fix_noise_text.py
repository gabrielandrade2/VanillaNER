import glob
import os
from argparse import ArgumentParser
from random import random

import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple


def parse_conll_ner_data(input_file: str, encoding: str = "utf-8"):
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


def decision(probability):
    return random() < probability


if __name__ == '__main__':

    parser = ArgumentParser()
    # parser.add_argument("--noise_ratio", type=float, required=True)
    # parser.add_argument("--noise_amount", type=float, required=True)
    args, _ = parser.parse_known_args()

    # noise_ratio = args.noise_ratio
    # noise_amount = args.noise_amount

    for file in tqdm(glob.glob('resources/noise_3/*/*'), desc='files'):
        print(file)
        iter = parse_conll_ner_data(file)
        filename = str(file).replace('/noise_3/', '/noise_fix_3/')
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
