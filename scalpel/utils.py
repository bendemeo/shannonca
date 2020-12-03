import scanpy as sc
import numpy as np


def ordered_matrixplot(d, n_genes=5, groups=None, **kwargs):
    """
    matrix plot of ranked groups, with columns ordered by score instead of abs(score).
    Separates up- from down-regulated genes better, resulting in visually-cleaner plots
    :param d:
    :param n_genes:
    :param kwargs:
    :return:
    """
    top_genes = np.stack([np.array(list(x)) for x in d.uns['rank_genes_groups']['names']][:n_genes])
    top_scores = np.stack([np.array(list(x)) for x in d.uns['rank_genes_groups']['scores']][:n_genes])

    # order top genes by actual score, not absolute value
    ordered_top_genes = np.take_along_axis(top_genes, np.argsort(-1 * top_scores, axis=0), axis=0)


    # print(ordered_top_genes)



    grouping_key = d.uns['rank_genes_groups']['params']['groupby']
    group_names = list(d.uns['rank_genes_groups']['names'].dtype.fields.keys())

    ordered_top_mapping = {group_names[i]: ordered_top_genes[:, i] for i in range(len(group_names))}


    if groups is not None:
        ordered_top_mapping = {k:v for k, v in ordered_top_mapping.items() if k in groups}
        print(ordered_top_mapping)


    sc.pl.matrixplot(d, var_names=ordered_top_mapping, groupby=grouping_key, **kwargs)