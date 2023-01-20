# Forked from Shogo Ujiie's IOB-Util module
# Available in https://github.com/ujiuji1259/IOB-util

"""Util function around iob (Shogo Ujiie ls)

This module contains iob util

Example:
    IOB decoding::

        convert_iob_to_dict(['私', 'は', '宇', '宙', '人'], ['O', 'O', 'B-C', 'I-C', 'I-C'])
        convert_iob_to_xml(['私', 'は', '宇', '宙', '人'], ['O', 'O', 'B-C', 'I-C', 'I-C'])

    IOB encoding::

        iobs = convert_xml_to_iob('私は<C value="N">宇宙人</C>', tag_lists=['C'], attr=['value'], tokenizer=list)
        print_iob(iobs)

Gabriel Andrade modifications:
    - Allow direct extraction from string containing entities annotated using xml tags (eg. <m-key state="executed">市販の薬</m-key>).
    - Better support to handle mismatching tags (eg. missing leading or trailing tag).
    - Allow support for tags using 'i' (e.g. 'm-key')
"""
from collections import deque

import lxml.etree as etree
from lxml.etree import XMLSyntaxError
from seqeval.metrics import accuracy_score, f1_score, precision_score, classification_report
from seqeval.scheme import IOB2

from util.list_utils import list_size


def split_tag(tag):
    if tag == "O":
        return tag, None
    else:
        try:
            t, l = tag.split('-', 1)
            return t, l
        except ValueError:
            return tag, ''


def unzip_iob(iobs):
    """Unzip IOB2

    unzip IOB2

    Args:
        iobs (List): [(token1, IOB2_1), (token2, IOB2_2), ...]

    Returns:
        List: [token1, token2, ...]
        List: [IOB2_1, ...]
    """

    tokens, labels = zip(*iobs)
    return list(tokens), list(labels)


def convert_iob_to_dict(tt, ii):
    """Convert iob to dict

    Convert tokens and IOB2 labels to dict format

    Args:
        tt (List): token list
        ii (List): IOB2 label list

    Returns:
        List: List of dict. Format = [{'span':(start_idx, end_idx), 'type': tag, 'word': word}]

    """
    assert len(tt) == len(ii), ''

    ii = ['O'] + ii + ['O']
    s_pos = -1
    word = ''
    result = []
    for idx in range(1, len(ii) - 1):
        prefix, tag = split_tag(ii[idx])
        if is_chunk_start(ii[idx - 1], ii[idx]):
            s_pos = idx - 1

        if s_pos != -1:
            word += tt[idx - 1]

        if is_chunk_end(ii[idx], ii[idx + 1]):
            result.append({'span': (s_pos, idx), 'type': tag, 'word': word})
            s_pos = -1
            word = ''

    return result


def convert_iob_taglist_to_dict(ii):
    """Convert iob tagslist to dict

    Convert tokens and IOB2 labels to dict format

    Args:
        ii (List): IOB2 label list

    Returns:
        List: List of dict. Format = [{'span':(start_idx, end_idx), 'type': tag}]

    """
    ii = ['O'] + ii + ['O']
    s_pos = -1
    result = []
    for idx in range(1, len(ii) - 1):
        prefix, tag = split_tag(ii[idx])
        if is_chunk_start(ii[idx - 1], ii[idx]):
            s_pos = idx - 1

        if is_chunk_end(ii[idx], ii[idx + 1]):
            result.append({'span': (s_pos, idx), 'type': tag})
            s_pos = -1

    return result


def convert_dict_to_xml(sent, dd):
    # Generate all text tags
    dd = sorted(dd, key=lambda x: x['span'][0])
    tags = []
    for d in dd:
        tags.append((d['span'][0], "<" + d['type'] + ">"))  # Start tag
        tags.append((d['span'][1], "</" + d['type'] + ">"))  # End tag
    tags = sorted(tags, key=lambda x: x[0])

    # Tag texts
    offset = 0
    for tag in tags:
        sent = sent[:tag[0] + offset] + tag[1] + sent[tag[0] + offset:]
        offset += len(tag[1])
    return sent


def convert_taglist_to_xml(sent, dd):
    # Generate all text tags
    dd = sorted(dd, key=lambda x: x[0])
    tags = []
    for d in dd:
        tags.append((d[0], "<" + d[2] + ">"))  # Start tag
        tags.append((d[1], "</" + d[2] + ">"))  # End tag
    tags = sorted(tags, key=lambda x: x[0])

    # Tag texts
    offset = 0
    for tag in tags:
        sent = sent[:tag[0] + offset] + tag[1] + sent[tag[0] + offset:]
        offset += len(tag[1])
    return sent


def convert_taglist_to_dict(taglist):
    dict_tags = []
    for tag in taglist:
        dict_tags.append({
            'span': [tag[0], tag[1]],
            'type': tag[2],
            'word': tag[3]
        })
    return dict_tags

def convert_iob_to_xml(tokens, iobs):
    """Convert iob to xml

    Convert tokens and IOB2 labels to xml format.

    Args:
        tokens (List): token list
        iobs (List): IOB2 label list

    Returns:
        str: Xml output.

    """
    dic = convert_iob_to_dict(tokens, iobs)
    return convert_dict_to_xml(''.join(tokens), dic)


def convert_list_iob_to_xml(list_tokens, list_iobs):
    texts = list()
    for tokens, iobs in zip(list_tokens, list_iobs):
        texts.append(convert_iob_to_xml(tokens, iobs))
    return '\n'.join(texts)


def convert_xml_to_taglist(sent, tag_list=None, attr=[], ignore_mismatch_tags=True):
    text = '<sent>' + sent + '</sent>'

    # Adding recover parameter allows handling missing tags.
    # It will reject closing tags with not start.
    # It will consider the start tag to span until the end of the sentence, if not close is found.
    parser = etree.XMLPullParser(['start', 'end'], recover=not ignore_mismatch_tags)
    parser.feed(text)

    ne_type = "O"
    ne_prefix = ""
    res = ""
    label = []
    tag_set = deque()
    s_pos = -1
    idx = 0
    word = ''

    for event, elem in parser.read_events():
        isuse = (tag_list is None or (tag_list is not None and elem.tag in tag_list))

        if event == 'start':
            # assert len(tag_set) < 2, "タグが入れ子になっています\n{}".format(sent)
            s_pos = idx

            if attr is not None and elem.attrib:
                attr_list = ''.join([v for k, v in elem.attrib.items() if k in attr])
            else:
                attr_list = ''

            word = elem.text if elem.text is not None else ""
            res += word
            idx += len(word)

            if elem.tag != 'sent' and isuse:
                label_list = [s_pos, idx, elem.tag + attr_list, word, elem.tag]
                tag_set.append(label_list)
                # label.append((s_pos, idx, elem.tag + attr_list, word))

        if event == 'end':
            if elem.tag != 'sent' and isuse and tag_set[-1][-1] == elem.tag:
                # and tag_set[-1] == elem.tag:
                label_list = tag_set.pop()
                label.append(tuple(label_list[:-1]))
                for tag in tag_set:
                    tag[1] = idx
                    tag[3] += word
            word = elem.tail if elem.tail is not None else ""
            res += word
            idx += len(word)

    return res, label

def convert_xml_to_dict(sent, tag_list=None, attr=[], ignore_mismatch_tags=True):
    res, label = convert_xml_to_taglist(sent, tag_list=tag_list, attr=attr, ignore_mismatch_tags=ignore_mismatch_tags)
    label_dict = []
    for tag in label:
        label_dict.append({
            "span": (int(tag[0]), int(tag[1])),
            "word": tag[3],
            "type": tag[2]
        })

    return res, label_dict

def convert_taglist_to_iob(sent, label, tokenizer=list):
    tokens = tokenizer(sent)
    results = []

    idx = 0
    i = 0
    j = 0

    nebegin = True

    while j < len(sent) and idx < len(label):
        k = j + len(tokens[i]) - 1
        if k < label[idx][0]:
            results.append((tokens[i], 'O'))
        elif label[idx][0] <= k and nebegin:
            results.append((tokens[i], 'B-' + label[idx][2]))
            nebegin = False
        else:
            results.append((tokens[i], 'I-' + label[idx][2]))

        j += len(tokens[i])
        i += 1

        while idx < len(label) and label[idx][1] <= j:
            idx += 1
            nebegin = True

    while i < len(tokens):
        results.append((tokens[i], 'O'))
        i += 1

    results = [i for i in results if not i[0] == ' ' or i[0] == '']
    return results


def convert_xml_text_to_iob(sent, tag_list=None, attr=None, tokenizer=list, ignore_mismatch_tags=True):
    """Convert xml to iob.

    Convert xml to IOB2 format. You can limit valid tag and attribute.

    Args:
        sent (str): Input xml string.
        tag_list (List): List of valid tag.
        attr (List): List of valid attribute.
        tokenizer (callable): Tokenize function. str->List
        ignore_mismatch_tags (bool): Should it try to recover if tags are missing?

    Returns:
        List (tuple): List of (token, IOB2 tag)
    """
    res, label = convert_xml_to_taglist(sent, tag_list=tag_list, attr=attr, ignore_mismatch_tags=ignore_mismatch_tags)
    iob = convert_taglist_to_iob(res, label, tokenizer=tokenizer)
    return [item for item in iob if item[0] != '\n']


def convert_xml_text_list_to_iob_list(texts, tag_list=None, attr=None, ignore_mismatch_tags=True,
                                      print_failed_sentences=False):
    """Convert a list of texts with xml tags into iob list format.

    :param texts: List of xml texts (List(str))
    :param tag_list: List of tags to be extracted (List(str))
    :param attr: List of tag attributes to be extracted (List(str))
    :param ignore_mismatch_tags: Should it try to recover if tags are missing?
    :param print_failed_sentences: Should it print the failed sentences for debug purposes?
    :return:
    """
    print("Converting xml to iob...")
    items = list()
    tags = list()
    dropped = list()
    i = 0
    for t in texts:
        sent = list()
        tag = list()
        try:
            iob = convert_xml_text_to_iob(t, tag_list, attr, ignore_mismatch_tags=ignore_mismatch_tags)
            # Convert tuples into lists
            for item in iob:
                if item[0] == ' ':
                    continue
                sent.append(item[0])
                tag.append(item[1])
            items.append(sent)
            tags.append(tag)
        except XMLSyntaxError as e:
            if print_failed_sentences:
                dropped.append(i)
                print("Skipping text with xml syntax error, id: " + str(i))
                print(t)
            elif not ignore_mismatch_tags:
                raise e
        i = i + 1
    if print_failed_sentences:
        return items, tags, dropped
    return items, tags


def evaluate_performance(original_labels, predict_labels):
    ##### Insanity check #####
    assert list_size(original_labels) == list_size(predict_labels)

    ###### Calculate perfromance metrics #####

    print('Accuracy: ' + str(accuracy_score(original_labels, predict_labels)))
    print('Precision: ' + str(precision_score(original_labels, predict_labels)))
    print('F1 score: ' + str(f1_score(original_labels, predict_labels)))
    # print(classification_report(original_labels, labels))
    print(classification_report(original_labels, predict_labels, mode='strict', scheme=IOB2))


def print_iob(iob):
    for t, l in iob:
        print(t + '\t' + l)


def is_chunk_end(tag, post_tag):
    prefix1, chunk_type1 = split_tag(tag)
    prefix2, chunk_type2 = split_tag(post_tag)

    if prefix1 == 'O':
        return False
    elif prefix2 == 'B':
        return True
    elif prefix2 == 'O':
        return prefix1 != 'O'

    return chunk_type1 != chunk_type2


def is_chunk_start(prev_tag, tag):
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'B':
        return True
    if prefix2 == 'O':
        return False
    if prefix1 == 'O' and prefix2 == 'I':
        return True

    return chunk_type1 != chunk_type2


def load_iob(fn, z=True):
    """Load IOB2 file.

    Load IOB2 file.

    Args:
        fn (str): File path of IOB2 file.
        z (bool): Output format. True means this returns [(token, IOB2)] and False means [token_list, IOB2_list]

    """
    with open(fn, 'r') as f:
        iobs = [lines.split('\n') for lines in f.read().split('\n\n') if lines != '']
        iobs = [[i.split('\t') for i in ii] for ii in iobs]

    if not z:
        iobs = [list(zip(*iob)) for iob in iobs]
        return [list(iob[0]) for iob in iobs], [list(iob[1]) for iob in iobs]

    return iobs


# Test code
if __name__ == '__main__':
    text = 'This is a <c><core>test</core></c> <a>string <core>containing</core> multiple</a> tags <d>stacked</d>.'
    untagged_text, tags = convert_xml_to_taglist(text)
    print(tags)
    convert_taglist_to_xml(untagged_text, tags)
    untagged_text, tags = convert_xml_to_taglist(text)
    print(tags)

