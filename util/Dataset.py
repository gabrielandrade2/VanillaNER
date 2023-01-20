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
