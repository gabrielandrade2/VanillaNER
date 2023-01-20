import numpy as np

def flatten(l):
    """Reduces the dimension of a list by 1.

    Parameters
    ----------
    l : list
        List with more than one dimension

    Returns
    -------
    list
        Flattened list
    """
    return [item for sublist in l for item in sublist]


def remove_prefix(labels):
    """Remove prefixes "B-"/"I-" from labels and only keep argument information

    Parameters
    ----------
    labels : list
        List of labels of the form {B|I}-{causal argument} or O

    Returns
    -------
    list
        List of labels of the form {causal argument | O}
    """
    return [l if l == "O" else l[2:] for l in labels]


def get_named_tuples(labels):
    """Get entities with positions from labels.

    Parameters
    ----------
    labels : list
        List of labels for each token

    Returns
    -------
    list
        List of dicts, dict contains type and position of entity
    """

    entities = []
    current_ent = []
    label_with_B = ""
    for pos, label in enumerate(labels):
        if label == "O":
            if current_ent:
                entities.append((label_with_B, current_ent))
            current_ent = []
            label_with_B = ""
            continue

        current_label_type = label[2:]

        # if new "B" is found, store label
        if label[0] == "B":
            if current_ent:
                entities.append((label_with_B, current_ent))

            current_ent = [pos]
            label_with_B = current_label_type

        # append only if I and same label
        if label[0] == 'I':
            if label_with_B == current_label_type:
                current_ent.append(pos)
            else:
                if current_ent:
                    entities.append((label_with_B, current_ent))
                current_ent = []
                label_with_B = ""

    # append leftover
    if len(current_ent) and label_with_B != "":
        entities.append((label_with_B, current_ent))

    entities = [{"type": typ, "pos": pos} for typ, pos in entities]

    return entities


def calc_precision(tp, fp):
    if (tp + fp) > 0:
        return tp / (tp + fp)
    else:
        return 0


def calc_recall(tp, fn):
    if (tp + fn) > 0:
        return tp / (tp + fn)
    else:
        return 0


def calc_f1(tp, fp, fn):
    if (tp + fn + fp) > 0:
        return tp / (tp + 0.5 * (fp + fn))
    else:
        return 0


def calculate_relaxed_metric(y_true, y_pred):
    """Calculates relaxed metric for prediction, where at least one token needs
    to be overlapping for tp.

    Parameters
    ----------
    y_true : list
        List of lists containing the ground truth labels
    y_pred : list
        List of lists containing the predicted labels

    Returns
    -------
    dict
        Dictionary with relaxed metrics
    """

    ent_types = (set(remove_prefix(flatten(y_true))) | set(remove_prefix(flatten(y_pred)))) - set(["O"])
    results = {}

    for ent_type in ent_types:
        results[ent_type] = {
            "tp": 0, "fp": 0, "fn": 0,
        }

    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        true_ents = get_named_tuples(y_true_sample)
        pred_ents = get_named_tuples(y_pred_sample)

        for ent_type in ent_types:
            true_ents_type = [ent for ent in true_ents if ent["type"] == ent_type]
            pred_ents_type = [ent for ent in pred_ents if ent["type"] == ent_type]

            true_matched = []
            pred_matched = []

            overlap_mat = np.zeros((len(true_ents_type), len(pred_ents_type)))

            for true_i, true_ent in enumerate(true_ents_type):
                for pred_i, pred_ent in enumerate(pred_ents_type):
                    overlap_mat[true_i, pred_i] = len(set(true_ent["pos"]) & set(pred_ent["pos"])) / len(
                        set(true_ent["pos"]) | set(pred_ent["pos"]))  # Jaccard index

            # get matches greedily
            while np.sum(overlap_mat > 0):
                best_score = overlap_mat.max()
                true_match, pred_match = np.where(overlap_mat == best_score)

                for pred_i, true_i in zip(pred_match, true_match):
                    if true_i in true_matched or pred_i in pred_matched: continue

                    true_matched.append(true_i)
                    pred_matched.append(pred_i)

                    overlap_mat[true_i, pred_i] = 0

                overlap_mat[true_match, pred_match] = 0

            results[ent_type]["tp"] += len(true_matched)
            results[ent_type]["fn"] += len(set(np.arange(len(true_ents_type))) - set(true_matched))
            results[ent_type]["fp"] += len(set(np.arange(len(pred_ents_type))) - set(pred_matched))

    for ent_type in ent_types:
        results[ent_type]["precision"] = calc_precision(tp=results[ent_type]["tp"], fp=results[ent_type]["fp"])
        results[ent_type]["recall"] = calc_recall(tp=results[ent_type]["tp"], fn=results[ent_type]["fn"])
        results[ent_type]["f1"] = calc_f1(tp=results[ent_type]["tp"], fp=results[ent_type]["fp"],
                                          fn=results[ent_type]["fn"])

    overall_result = {}
    overall_result["tp"] = sum([results[ent_types]["tp"] for ent_types in ent_types])
    overall_result["fn"] = sum([results[ent_types]["fn"] for ent_types in ent_types])
    overall_result["fp"] = sum([results[ent_types]["fp"] for ent_types in ent_types])

    overall_result["precision"] = calc_precision(tp=overall_result["tp"], fp=overall_result["fp"])
    overall_result["recall"] = calc_recall(tp=overall_result["tp"], fn=overall_result["fn"])
    overall_result["f1"] = calc_f1(tp=overall_result["tp"], fp=overall_result["fp"], fn=overall_result["fn"])

    results["overall"] = overall_result
    return results
