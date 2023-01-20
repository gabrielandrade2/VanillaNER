import shutil

def __read_gold_file(gold_file):
    gold = []
    use_document = False
    with open(gold_file, 'r') as f:
        for line in f:
            if '-DOCSTART-' in line:
                if 'testb' in line:
                    use_document = True
                else:
                    use_document = False

            if use_document:
                splits = line.strip().split('\t')
                if len(splits) >= 3:
                    gold.append(splits[3:])
                else:
                    gold.append([])
    return gold


def add_gold_entity_to_NER_iob_output(input_file, gold_file):
    # Read only testb documents from goldfile
    gold = __read_gold_file(gold_file)

    print("Adding gold entities to NER IOB output")
    processed_lines = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            splits = line.strip().split('\t')

            gold_entities = gold[i]
            if len(gold_entities) != 0:
                splits.extend(gold_entities)
            processed_lines.append(splits)

    for i, line in enumerate(processed_lines):
        if len(line) != 2:
            continue
        if line[1] == 'B':
            j = 1
            while i + j < len(processed_lines)-1 and len(processed_lines[i + j]) >= 2 and processed_lines[i + j][1] == 'I':
                if len(processed_lines[i + j]) > 2:
                    line.extend(processed_lines[i + j][2:])
                    break;
                j += 1

    output_file = input_file.replace('output.iob', 'output_gold.iob')
    with open(output_file, 'w') as g:
        for line in processed_lines:
            g.write('\t'.join(line) + '\n')
    shutil.move(output_file, input_file)
