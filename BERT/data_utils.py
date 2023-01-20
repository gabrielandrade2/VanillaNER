def load_dataset(path):
    with open(path, 'r') as f:
        lines = f.read()

    lines = lines.split("\n\n")
    lines = [[token for token in l.split("\n") if token != ""] for l in lines if l != ""]

    data = []
    label = []
    for line in lines:
        sent = []
        sent_label = []
        for l in line:
            token, tag = l.split("\t")
            sent_label.append(tag)

        data.append(sent)
        label.append(sent_label)

    return data, label


def create_vocab(data):
    vocab = {}
    vocab["[PAD]"] = len(vocab)
    vocab["[UNK]"] = len(vocab)

    for d in data:
        for token in d:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def create_label_vocab(label):
    vocab = {}
    vocab["[PAD]"] = len(vocab)

    for l in label:
        for token in l:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def sent2input(sent, vocab):
    return [vocab[token] if token in vocab else vocab["[PAD]"] for token in sent]


def data2input(data, vocab):
    return [sent2input(sent, vocab) for sent in data]


def pad_sentence(sent, length, pad_value=0):
    return sent + [pad_value] * (length - len(sent)) if len(sent) <= length else sent[:length]


def pad_sequence(seq, issort, max_length=512, pad_value=0):
    length = len(seq[0]) if issort else len(sorted(seq, key=lambda x: len(x), reverse=True)[0])
    max_length = min(length, max_length)
    return [pad_sentence(s, max_length, pad_value) for s in seq]


class Batch(object):
    def __init__(self, sentence, label, batch_size=8, pad_value=0, max_size=512, sort=True):
        self.batch_size = batch_size
        self.pad_value = pad_value
        self.max_size = max_size
        self.sort = sort
        if self.sort:
            self.data = sorted(zip(sentence, label), key=lambda x: len(x[0]), reverse=True)
        else:
            self.data = list(zip(sentence, label))

    def get_sentences(self):
        return [d[0] for d in self.data]

    def get_labels(self):
        return [d[1] for d in self.data]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            s_pos = i
            e_pos = min(i + self.batch_size, len(self.data))

            x = [d[0] for d in self.data[s_pos:e_pos]]
            l = [d[1] for d in self.data[s_pos:e_pos]]
            length = [len(d[0]) for d in self.data[s_pos:e_pos]]
            x = pad_sequence(x, self.sort, max_length=self.max_size, pad_value=self.pad_value)
            l = pad_sequence(l, self.sort, max_length=self.max_size, pad_value=self.pad_value)

            yield x, l, length


class Mydataset(object):
    def __init__(self, path, vocab=None, label_vocab=None, batch_size=8):
        self.data, self.label = load_dataset(path)
        self.vocab = vocab if vocab is not None else create_vocab(self.data)
        self.label_vocab = label_vocab if label_vocab is not None else create_label_vocab(self.label)
        self.x, self.l = data2input(self.data, self.vocab), data2input(self.label, self.label_vocab)
        self.batch_size = batch_size

    def get_vocab(self):
        return self.vocab

    def get_label_vocab(self):
        return self.label_vocab

    def __iter__(self):
        data = zip(self.x, self.l)
        data = sorted(data, key=lambda x: len(x[0]), reverse=True)

        for i in range(0, len(data), self.batch_size):
            s_pos = i
            e_pos = min(i + self.batch_size, len(data))

            x = [d[0] for d in data[s_pos:e_pos]]
            l = [d[1] for d in data[s_pos:e_pos]]
            length = [len(d[0]) for d in data[s_pos:e_pos]]
            x = pad_sequence(x, pad_value=self.vocab["[PAD]"])
            l = pad_sequence(l, pad_value=self.label_vocab["[PAD]"])

            yield x, l, length

