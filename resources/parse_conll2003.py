filenames = ['eng.train', 'eng.testa', 'eng.testb']

for filename in filenames:
    fIn = open(filename, 'r')
    fOut = open(filename + '.iob', 'w')

    for line in fIn:
        if line.startswith('-DOCSTART-'):
            lastChunk = 'O'
            lastNER = 'O'
            continue

        if len(line.strip()) == 0:
            lastChunk = 'O'
            lastNER = 'O'
            fOut.write("\n")
            continue

        splits = line.strip().split()

        chunk = splits[2]
        ner = splits[3]

        if chunk[0] == 'I':
            if chunk[1:] != lastChunk[1:]:
                chunk = 'B' + chunk[1:]

        if ner[0] == 'I':
            if ner[1:] != lastNER[1:]:
                ner = 'B' + ner[1:]

        splits[2] = chunk
        splits[3] = ner

        fOut.write("\t".join(splits))
        fOut.write("\n")

        lastChunk = chunk
        lastNER = ner

print("--DONE--")