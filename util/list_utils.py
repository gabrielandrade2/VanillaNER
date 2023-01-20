def flatten_list(l: list):
    return [item for sublist in l for item in sublist]


def list_size(l: list):
    return sum([len(t) for t in l])


def dict_mean(dict_list: list):
    mean_dict = {}
    for key in dict_list[0].keys():
        try:
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
        except Exception:
            pass
    return mean_dict
