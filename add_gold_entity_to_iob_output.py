import glob

from tqdm import tqdm

if __name__ == '__main__':
    gold_file = '/Users/gabriel-he/PycharmProjects/SimpleNER/resources/AIDA-YAGO2-dataset-testbonly.tsv'
    input_folder = '/Users/gabriel-he/PycharmProjects/SimpleNER/results/new_noise_fix_batch_16_e-6'

    # Read gold file
    gold = []
    with open(gold_file, 'r') as f:
        for line in f:
            splits = line.strip().split('\t')
            if len(splits) >= 3:
                gold.append(splits[3:])
            else:
                gold.append([])

    # Read input file
    for input_file in tqdm(glob.glob(input_folder + '/**/testbout.txt', recursive=True)):
        # print(input_file)
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

        with open(input_file.replace('testbout', 'testbout_gold'), 'w') as g:
            for line in processed_lines:
                g.write('\t'.join(line) + '\n')

