from pprint import pprint

from util import iob_util


def score(gold, predicted, output_dict=None, print_results=False):
    score = 0
    matched = set()
    results = dict.fromkeys(
        ["exact_match", "exceeding_match", "exceeding_match_overlap", "partial_match", "partial_match_overlap",
         "missing_match", "incorrect_match"], 0)

    for entity in gold:
        found_match = False
        gold_start, gold_end = entity['span']
        gold_tag = entity['type']
        for i in range(len(predicted)):
            pred_start, pred_end = predicted[i]['span']
            pred_tag = predicted[i]['type']

            if found_match:
                break

            # Skip different tags
            if gold_tag != pred_tag:
                continue

            # Skip prediction tags that came before
            if gold_start >= pred_end:
                continue

            # We can stop searching now
            elif gold_end <= pred_start:
                break

            # Exact match
            elif pred_start == gold_start and pred_end == gold_end:
                if not found_match:
                    score += 1
                    results["exact_match"] += 1
                    matched.add(predicted[i]['span'])
                    found_match = True

            # Exceeding match, but not overlap next entity
            elif pred_start <= gold_start and pred_end >= gold_end:

                # Check for overlap
                if i + 1 < len(gold):
                    if not pred_end < gold[i + 1]['span'][0]:  # next gold start
                        if not found_match:
                            score += 0
                            results["exceeding_match_overlap"] += 1
                            matched.add(predicted[i]['span'])
                            found_match = True
                            continue

                if not found_match:
                    score += 0.5
                    results["exceeding_match"] += 1
                    matched.add(predicted[i]['span'])
                    found_match = True

            # Partial match
            elif pred_start >= gold_start and pred_end <= gold_end:
                if not found_match:
                    score += 0.5
                    results["partial_match"] += 1
                    matched.add(predicted[i]['span'])
                    found_match = True

            elif (pred_start >= gold_start and pred_end >= gold_end) or (
                    pred_start <= gold_start and pred_end <= gold_end):

                # Check for overlap
                if i + 1 < len(gold):
                    if not pred_end < gold[i + 1]['span'][0]:  # next gold start
                        if not found_match:
                            score += 0
                            results["partial_match_overlap"] += 1
                            matched.add(predicted[i]['span'])
                            found_match = True
                            continue

                if not found_match:
                    score += 0.5
                    results["partial_match"] += 1
                    matched.add(predicted[i]['span'])
                    found_match = True

            else:
                raise Exception("Insanity-check, should not be here!")

        if not found_match:
            score += 0
            results["missing_match"] += 1

    incorrect_match = len(predicted) - len(matched)
    score += 0
    results["incorrect_match"] = incorrect_match
    if print_results:
        pprint(results)
    if isinstance(output_dict, dict):
        for key in results:
            output_dict[key] = results[key]
    if len(gold):
        return score / (len(gold))
    return float(score)


def score_from_iob(gold, predicted, output_dict=None, print_results=False):
    # gold = list_utils.flatten_list(gold)
    # predicted = list_utils.flatten_list(predicted)
    gold = iob_util.convert_iob_taglist_to_dict(gold)
    gold = sorted(gold, key=lambda x: x['span'][0])
    predicted = iob_util.convert_iob_taglist_to_dict(predicted)
    predicted = sorted(predicted, key=lambda x: x['span'][0])

    return score(gold, predicted, output_dict, print_results)


def score_from_span(gold, predicted, output_dict=None, print_results=False):
    pass


if __name__ == '__main__':
    O = 'O'
    B = 'B'
    I = 'I'

    m = '-m'
    d = '-d'

    gold = [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, I, I, I, O, O, O, B, I, I, I, I, I, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, O, B, I, O, O, O, O, B, I, I, I, I, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, O, O, O, O, O, B, I, O, O, O, O, O, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, O, B, I, O, B, I, O, B, I, O, B, I, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, I, I, I, I, I, I, I, I, I, I, I, I, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, I, I, I, I, I, O, O, B, I, I, I, I, I, I]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, O, O, B, I, I, O, O, O, O, O, B, I, I, O, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, I, I, I, I, I, I, I, I, I, O, B, I, I, I]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    gold = [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]
    test = [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    gold = [O, O, B, I, I, I, O, O, O, O, B, I, I, I, O, O, O]
    test = [O, B, I, O, O, B, I, O, O, B, I, O, O, B, I, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')

    gold = [O, O, B + m, I + m, I + m, I + m, O, O, O, O, B + d, I + d, I + d, I + d, I + d, O, O]
    test = [O, O, B + m, I + m, I + m, I + m, O, O, O, O, B + m, I + m, I + m, I + m, I + m, O, O]
    print(gold)
    print(test)
    print(score_from_iob(gold, test))
    print('\n')
