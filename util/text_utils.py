import re
from abc import abstractmethod, ABC


def preprocessing(texts, remove_core_tag=True):
    """Preprocessing steps for strings.
    Strip strings, remove <core> tags (for now).

    :param texts: List of strings to be processed.
    :param remove_core_tag: Should remove core tag from the texts?
    :return: The list of processed texts.
    """

    processed_texts = list()
    for text in texts:
        if remove_core_tag:
            # Remove all <core> and </core> for now
            text = text.replace('<core>', '')
            text = text.replace('</core>', '')

        # Remove all \n from the beginning and end of the sentences
        text = text.strip()
        processed_texts.append(text)
    return processed_texts


def split_sentences(texts, return_flat_list=True):
    """Given a list of strings, split them into sentences and join everything together into a flat list containing all
     the sentences.

    :param texts: List of strings to be processed.
    :param return_flat_list: If True return a flat list with all the sentences, otherwise a list of lists.
    :return: The list of split sentences.
    """
    print("Splitting sentences...")
    processed_texts = list()
    for text in texts:
        processed_text = re.split(
            "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s\n*|(?<=[^A-zＡ-ｚ0-9０-９ ].)(?<=[。．.?？!！])(?![\.」])\n*", text)
        # processed_text = re.split("(? <=[。?？!！])")  # In case only a simple regex is necessary
        processed_text = [x.strip() for x in processed_text]
        processed_text = [x for x in processed_text if x != '']
        if return_flat_list:
            processed_texts.extend(processed_text)
        else:
            processed_texts.append(processed_text)
    return processed_texts


def exclude_long_sentences(max_length: int, sentences: list, tags: list):
    print(len(sentences))
    tmp_s = []
    tmp_t = []
    for s, t in zip(sentences, tags):
        if len(s) <= max_length:
            tmp_s.append(s)
            tmp_t.append(t)
    sentences = tmp_s
    tags = tmp_t
    print(len(sentences))
    return sentences, tags


def tag_matches(text, tags, tag_type):
    """ Add HTML tags to a text.

    :param text: The string that will have the tags added.
    :param tags: The tuple containing start/end position of the tags in the referring string.
    :param tag_type: The tag to be added (without <>).
    :return:
    """
    tags = sorted(tags)
    start_tag = "<" + tag_type + ">"
    end_tag = "</" + tag_type + ">"
    offset = len(start_tag) + len(end_tag)
    total_offset = 0
    for start, end, entry in tags:
        start += total_offset
        end += total_offset
        tagged = start_tag + text[start:end] + end_tag
        text = text[:start] + tagged + text[end:]
        total_offset += offset
    return text


def remove_tags(text, tag_list=None):
    """ Removes HTML tags from a text.

    :param text: The string that will have the tags removed.
    :param tag_list: The list of tags that should be removed. If None, all tags will be removed.
    """
    if tag_list:
        for tag in tag_list:
            regex = re.compile(r'<\/?{}>'.format(tag))
            text = re.sub(regex, '', text)
    else:
        text = re.sub('<[^<>]*>', '', text)
    return text


def findstem(arr):
    arr = sorted(arr, key=lambda x: len(x))

    # Determine size of the array
    n = len(arr)

    # Take first word from array
    # as reference
    s = arr[0]
    l = len(s)

    res = ""

    for i in range(l):
        for j in range(i + 1, l + 1):

            # generating all possible substrings
            # of our reference string arr[0] i.e s
            stem = s[i:j]
            k = 1
            for k in range(1, n):

                # Check if the generated stem is
                # common to all words
                if stem not in arr[k]:
                    break

            # If current substring is present in
            # all strings and its length is greater
            # than current result
            if (k + 1 == n and len(res) < len(stem)):
                res = stem

    return res


class EntityNormalizer(ABC):

    @abstractmethod
    def normalize(self, term):
        pass

    def normalize_list(self, terms):
        normalized_list = list()
        score_list = list()
        for term in terms:
            normalized_term, score = self.normalize(term)
            normalized_list.append(normalized_term)
            score_list.append(score)

        return normalized_list, score_list


class DrugNameMatcher(ABC):

    @abstractmethod
    def match(self, text):
        pass

    @staticmethod
    def exact_match(text1, text2, ignore=list()):
        ignore = iter(sorted(ignore))
        start = 0
        max = len(text1)
        length = len(text2)
        out = list()

        while True:
            item = next(ignore, None)
            if item:
                end = item[0]
            else:
                end = max

            while True:
                start = text1.find(text2, start, end)

                # Found nothing
                if start == -1:
                    # Reached the end
                    if end == max:
                        return out
                    # Jump over next ignore item
                    else:
                        start = item[1]
                    break

                # Found something
                else:
                    out.append((start, start + length, text2))
                    start += length
