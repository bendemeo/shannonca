from sklearn.metrics import f1_score
import numpy as np
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# def cluster_f1_dict(labels, true):
#     labs = np.unique(labels)
#     true_labs = np.unique(true)
#
#     mapping_dict = {}
#     for lab in labs:
#         overlap_scores = []
#         for tlab in true_labs:
#             overlap = np.sum(np.logical_and(labels == lab, true_labs == tlab))/np.sum(labels == lab)
#             overlap_scores.append(overlap)
#         mapping_dict[lab] =


def cluster_f1_best(labels, true, return_labs=False):
    # 1. identify all clusters intersecting the target population
    labs = np.unique(labels)
    overlapping_labs = []
    subsets = []
    for lab in labs:
        overlap = (np.logical_and(labels == lab, true))
        if np.sum(overlap) == np.sum(labels == lab):  # strict subset; add automatically
            subsets.append(lab)
        elif np.sum(np.logical_and(labels == lab, true)) > 0:
            overlapping_labs.append(lab)

    # print('{} overlapping clusters'.format(len(overlapping_labs)))

    sets = list(powerset(overlapping_labs))
    print('hi')
    print(len(list(sets)))
    print('{} combinations to try'.format(len(list(sets))))

    max_set = subsets
    max_f1 = None
    for candidate in list(sets):
        s = tuple(subsets) + candidate
        # print(s)
        if len(s) == 0 and len(subsets) == 0:  # skip empty set
            continue
        if len(s) > 2:
            union = np.logical_or.reduce(tuple([labels == x for x in s]))
        else:
            union = np.array(labels == s[0])
        f1 = f1_score(true, union)

        if max_f1 is None or f1 > max_f1:
            max_f1 = f1
            max_set = s
    if return_labs:
        return (max_f1, max_set)
    else:
        return (max_f1)


def cluster_f1_better(labels, true, return_labs=False):
    labs = np.unique(labels)
    overlapping_labs = []
    subsets = []
    max_indicator = np.zeros(len(labels))
    max_f1 = 0
    max_clusts = []

    lab_scores = []
    #order sets by (true overlap) vs. (false overlap) proportion
    for lab in labs:
        tp = np.sum(np.logical_and(labels == lab, true))
        fp = np.sum(np.logical_and(labels == lab, np.logical_not(true)))

        score = tp/(tp+fp)
        lab_scores.append((score, lab))

    ordered_labs = [x[1] for x in sorted(lab_scores, reverse=True)] # labs ordered by true pos proportion

    for lab in ordered_labs:
        indicator = max_indicator + (labels == lab)
        f1 = f1_score(true, indicator)
        if f1 >= max_f1:
            max_indicator = indicator
            max_clusts.append(lab)
            max_f1 = f1_score(true, max_indicator)
        else:
            break

    if return_labs:
        return (max_f1, max_clusts)
    else:
        return max_f1



def cluster_f1(labels, true, return_labs=False):
    # given cluster labels, compute the best F1 score obtained by a combination of these clusters

    max_f1 = 0
    max_clusts = []
    labels = np.array(labels)

    # find cluster with highest F1 on its own
    labs = np.unique(labels)
    max_indicator = np.zeros(len(labels))
    for lab in labs:  # add all clusters that are subsets of true pop
        if np.sum(np.logical_and(labels == lab, true)) == np.sum(labels == lab):
            max_clusts.append(lab)
            max_indicator += (labels == lab)
            max_f1 = f1_score(true, max_indicator)

    # for lab in labs:
    #     indicator = (labels == lab)
    #     f1 = f1_score(true, indicator)
    #     if f1 > max_f1:
    #         max_clusts = [lab]
    #         max_indicator = indicator
    #         max_f1 = f1

    static = False  # did max_f1 change after cycling through the labs?
    while not static:
        static = True
        for lab in labs:
            if lab in max_clusts:
                continue

            indicator = max_indicator + (labels == lab)
            f1 = f1_score(true, indicator)
            if f1 > max_f1:
                static = False
                max_f1 = f1
                max_indicator = indicator
                max_clusts += [lab]

    if return_labs:
        return (max_f1, max_clusts)
    else:
        return max_f1


def ranking_f1(rankings, true, return_idxs=False):
    order = np.array(list(reversed(np.argsort(rankings))))
    f1s = np.zeros(len(rankings))

    for i in range(len(rankings)):
        # print('{}/{}'.format(i, len(rankings)), end='\r')
        indicator = np.zeros(len(rankings))
        indicator[order[:i]] = 1
        f1s[i] = f1_score(true, indicator)
    if return_idxs:
        return (np.max(f1s), indicator)

    return np.max(f1s)
