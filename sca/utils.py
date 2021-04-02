import numpy as np


def metagene_loadings(data, n_genes=10, rankby_abs=False, key='sca'):
    """
    return the top-loaded genes for each Scalpel metagene
    :param data: AnnData or dict output from reduce_scanpy or reduce. Must contain loading data.
    :param n_genes: Number of top-loaded genes to record
    :param rankby_abs: If True, record the top genes by absolute value.
    :param key: Namespace of Scalpel data, if data is a scanpy object. Will look under data.varm[key+'_loadings'].
    :return: Dictionary mapping [sca component number]:{[top gene 1]:loading, [top_gene_2]:loading... [top_gene_k]:loading}
    """

    if type(data) is dict:
        loadings = data['loadings']
        var_names = np.array(list(range(loadings.shape[0])))
    else:
        loadings = data.varm[key+'_loadings']
        var_names = np.array(data.var_names)

    loadings /= np.sum(np.abs(loadings), axis=0) #normalize to percent contribution
    loadings *= loadings.shape[0] # fold enrichment over expected

    if rankby_abs:
        idxs = np.transpose(np.argsort(np.abs(loadings), axis=0)[(-1*n_genes):,:]).tolist()
    else:
        idxs = np.transpose(np.argsort(loadings, axis=0)[(-1 * n_genes):,:]).tolist()

    genes = [var_names[np.array(x)] for x in idxs]
    scores = [loadings[x,i] for i, x in enumerate(idxs)]


    result = list(zip(genes, scores))
    result = {i:{'genes':k[np.argsort(-1*v)], 'scores':v[np.argsort(-1*v)]} for i,(k,v) in enumerate(result)}
    return result
