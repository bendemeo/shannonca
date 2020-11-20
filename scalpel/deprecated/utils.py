from fbpca import pca
import scanpy as sc
import numpy as np
import warnings
from scipy.spatial.distance import cdist, euclidean

def add_graph(adata, G, name='neighbors_custom'):
    """a bit hacky; register G as a nearest neighbor graph in the indicated namespace"""

    adata.obsp[name+'_connectivities'] = G
    nbrs = G.sum(axis=1)[0]
    adata.uns[name] = {'connectivities_key': name+'_connectivities', 'distances_key': name+'_connectivities',
                       'params':{'n_neighbors':nbrs, 'method': name}}


def fast_pca(X, n_pcs=50):
    """run Facebook's fast PCA algorithm on X"""

    U,s,Vt = pca(X, k=n_pcs)
    return U*s

def umap_adata(adata, rep_key='umap', **kwargs):
    """slight enhancement of sc.tl.umap, enabling arbitrary .obsm keys"""
    sc.tl.umap(adata, **kwargs)
    adata.obsm['X_{}'.format(rep_key)] = adata.obsm['X_umap']

def viz_adata(adata, n_pcs = 50, n_neighbors=15, rerun=False, rep='umap', **kwargs):


    if rerun or 'X_{}'.format(rep) not in adata.obsm.keys():

        if rep != 'umap':
            warnings.warn("don't know how to generate rep {}; making UMAP".format(rep))
            rep = 'umap'

        n_pcs = min([n_pcs, adata.shape[1]-1])
        sc.tl.pca(adata, n_comps=n_pcs)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca')
        sc.tl.umap(adata)

    if rep == 'umap':
        ax = sc.pl.umap(adata, **kwargs)
        return ax

    else:
        sc.pl.embedding(basis=rep, adata=adata, neighbors_key='X_{}'.format(rep), **kwargs)


def binarize(adata):
    adata.X = (adata.X>0).astype('float')







def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1