"""functions for evaluating embeddings and neighbor networks"""
import scanpy as sc
import numpy as np
import pandas as pd


def fidelity(adata, name, groupby='label'):
    """look in .obsp[name_connectivities] for connectivity data and see how well it preserves the given labelling"""

    labels = adata.obs[groupby].values

    adj = (adata.obsp[name+'_'*(len(name)>0)+'connectivities']>0)
    edgelist = zip(*adj.nonzero())

    good_edges=0
    bad_edges=0
    for i,j in edgelist:
        if labels[i] == labels[j]:
            good_edges += 1
        else:
            bad_edges += 1


    return float(good_edges)/float(good_edges + bad_edges)


def group_fidelity(adata, name, groupby='label', pairwise=False):
    """matrix containing the fraction of edges that stay within each group
    if pairwise=True, returns the fraction of edges between each pair of groups
    i.e. position (i,j) is the proportion of edges that land in group j, among
    all edges originating in group i"""

    classes = adata.obs[groupby].values
    G = (adata.obsp[name+'_'*(len(name)>0)+'connectivities']>0)

    class_list = np.unique(classes)

    result = np.zeros((len(class_list), len(class_list)))

    for i, c in enumerate(class_list):
        # print(i)
        # print(c)
        inds = np.where(np.array(classes) == c)[0]

        G_sub = G[inds, :]  # only look at vertices in c
        class_freqs = [0] * len(class_list)
        for j in range(len(inds)):
            row = G_sub[j, :].todense().tolist()[0]
            row = np.array(row).astype(int)

            # print(G_sub[j,:].tolist()[0])
            nbr_inds = np.where(row > 0)[0]
            # print(nbr_inds)
            nbr_classes = np.array([classes[x] for x in nbr_inds])

            for k, c2 in enumerate(class_list):
                class_freqs[k] += np.sum(nbr_classes == c2)

        # print(class_freqs)
        result[i, :] = class_freqs

    result = result / result.sum(axis=1)[:, None]

    result = pd.DataFrame(result)
    result.columns = class_list
    result.index = class_list

    if pairwise:
        return (result)
    else:
        diags = np.diag(result.values)
        return pd.DataFrame({'cluster':class_list, 'fidelity':diags, 'method':name})



def plot_umap(adata, name, n_neighbors=10, rerun=False, **kwargs):
    """looks in .uns[name] for neighbors info, and uses that to store and plot a UMAP"""

    if name not in adata.uns or rerun:
        sc.pp.neighbors(adata, use_rep='X_'+name, key_added=name, n_neighbors=n_neighbors)

    if 'X_'+name+'_umap' not in adata.obsm or rerun:
        sc.tl.umap(adata, neighbors_key=name)
        adata.obsm['X_{}_umap'.format(name)] = adata.obsm['X_umap']

    sc.pl.embedding(adata, basis='{}_umap'.format(name), neighbors_key=name, **kwargs)







def good_bad_edges(adata, name, groupby='label'):

    labels = adata.obs[groupby].values

    adj = (adata.obsp[name + '_' * (len(name) > 0) + 'connectivities'] > 0)
    edgelist = zip(*adj.nonzero())

    good_edges = []
    bad_edges = []

    for i, j in edgelist:
        if labels[i] == labels[j]:
            good_edges.append((i,j))
        else:
            bad_edges.append((i,j))

    return (good_edges, bad_edges)