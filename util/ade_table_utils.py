import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from util import iob_util


def convert_labels_to_dict(sentences, labels):
    ne_dict = list()
    for sent, label in zip(sentences, labels):
        ne_dict.extend(iob_util.convert_iob_to_dict(sent, label))
    return ne_dict


def normalize_entities(named_entities, normalization_model):
    normalized_entities = list()
    for entry in named_entities:
        entry['normalized_word'], _ = normalization_model.normalize(entry['word'])
        normalized_entities.append(entry)
    return normalized_entities


def consolidate_table_data(drug, output_dict, ne_dict):
    if drug in output_dict:
        drug_dict = output_dict[drug]
    else:
        drug_dict = {}

    for named_entity in ne_dict:
        word = named_entity['normalized_word']
        if word in drug_dict:
            count = drug_dict[word] + 1
        else:
            count = 1
        drug_dict[word] = count
    output_dict[drug] = drug_dict
    return output_dict


def table_post_process(table):
    # Order drugs by number of ADE events
    table['sum_col'] = table.sum(axis=1)
    table.sort_values('sum_col', axis=0, ascending=False, inplace=True)
    table.drop(columns=["sum_col"], inplace=True)
    table = table[:50]

    # Order ADE by numer of events
    table = table[table.sum(0).sort_values(ascending=False)[:50].index]

    return table


def generate_heatmap(table):
    mpl.rc('font', family="Hiragino Sans")
    sns.set(font='Hiragino Sans')
    sns.color_palette("YlOrBr", as_cmap=True)
    plt.figure(figsize=(20, 20))
    heatmap = sns.heatmap(table)
    heatmap.figure.tight_layout()
    fig = heatmap.get_figure()
    fig.savefig("out.png")
    plt.show()
