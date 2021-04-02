from .dimred import reduce, reduce_scanpy
import numpy as np
import pandas as pd

def correct(Xs, feat_names=None, n_features=1000, **kwargs):
    if feat_names is None:
        feat_names = range(Xs[0].shape[1])

    sigs = np.zeros((len(Xs), Xs[0].shape[1]))
    for i,X in enumerate(Xs):
        scores = reduce(X, keep_scores=True, **kwargs)['scores']
        gene_sigs = np.array(abs(scores).mean(axis=0)).flatten()
        #print(gene_sigs)
        sigs[i,:] = gene_sigs

    gene_scores = np.mean(np.abs(sigs), axis=0)

    top_feats = np.argsort(gene_scores)[(-1*n_features):]

    top_names = np.array(feat_names)[top_feats]

    corrected_Xs = [X[:, top_feats] for X in Xs]

    return (corrected_Xs, top_names)


def correct_scanpy(adata, batch_key='batch', n_features=1000, **kwargs):
    gene_sigs_dict = {}
    for i,batch in enumerate(np.unique(adata.obs[batch_key].tolist())):
        #print(batch)
        abatch = adata[adata.obs[batch_key].values.astype('str')==str(batch),:].copy()
        reduce_scanpy(abatch, keep_scores=True, keep_loadings=False, **kwargs)
        gene_sigs = np.array(np.mean(np.abs(abatch.layers['scalpel_score'].todense()), axis=0)).flatten()
        gene_sigs_dict[batch] = gene_sigs

    gene_sig_df = pd.DataFrame(gene_sigs_dict)
    gene_scores = np.mean(gene_sig_df.values, axis=1)
    most_sig_genes = adata.var.index[np.argsort(gene_scores)[(-1*n_features):]]

    adata_corrected = adata[:,most_sig_genes].copy()
    return (adata_corrected, most_sig_genes)

#def correct_scanpy(adata, batch_key):