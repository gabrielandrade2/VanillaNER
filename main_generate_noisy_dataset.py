from util.noise import dataset_noise

'''
Script used to generate boundary-expanded variants of the AIDA-CoNLL-YAGO dataset.
'''
if __name__ == '__main__':
    # Only one word to each side
    dataset_noise.generate_noisy_datasets("resources/AIDA-YAGO2-dataset.tsv", [1])
