import numpy as np
#from .score import info_score
from scipy.sparse import issparse, csr_matrix
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

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
        loadings = data.varm[key + '_loadings']
        var_names = np.array(data.var_names)

    loadings /= np.sum(np.abs(loadings), axis=0)  # normalize to percent contribution
    loadings *= loadings.shape[0]  # fold enrichment over expected

    if rankby_abs:
        idxs = np.transpose(np.argsort(np.abs(loadings), axis=0)[(-1 * n_genes):, :]).tolist()
    else:
        idxs = np.transpose(np.argsort(loadings, axis=0)[(-1 * n_genes):, :]).tolist()

    genes = [var_names[np.array(x)] for x in idxs]
    scores = [loadings[x, i] for i, x in enumerate(idxs)]

    result = list(zip(genes, scores))
    result = {i: {'genes': k[np.argsort(-1 * v)], 'scores': v[np.argsort(-1 * v)]} for i, (k, v) in enumerate(result)}
    return result

def scramble_genes(X):
    if issparse(X):
        Xin = X.todense()
    else:
        Xin=X

    # from https://stackoverflow.com/questions/27486677/best-way-to-permute-contents-of-each-column-in-numpy
    ix_i = np.random.sample(Xin.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(Xin.shape[1]), (Xin.shape[0], 1))

    Xin =  Xin[ix_i, ix_j]

    if issparse(X):
        return csr_matrix(Xin)
    else:
        return(Xin)



def binomial_background(X, nbhd_size=15, n_pcs=20, metric='cosine'):
    rep = sc.tl.pca(X, n_comps=n_pcs)

    nn = NearestNeighbors(n_neighbors=nbhd_size, metric=metric)
    nn.fit(rep)

    nbhds = nn.kneighbors(return_distance=False)

def make_umaps(adata, obsm_keys, n_neighbors=15, metric='cosine', **kwargs):
    for k in obsm_keys:
        print(k)
        sc.pp.neighbors(adata, use_rep='X_'+k, key_added=k, metric=metric, n_neighbors=n_neighbors)
        sc.tl.umap(adata, neighbors_key=k, **kwargs)
        adata.obsm['X_umap_{}'.format(k)] = adata.obsm['X_umap']


def plot_umaps(adata, umap_keys, plot_size=3, **kwargs):
    fig, axs = plt.subplots(1, len(umap_keys))
    if len(umap_keys)==1:
        axs = [axs]
    fig.set_size_inches(plot_size*len(umap_keys), plot_size)

    for i,k in enumerate(umap_keys):

        sc.pl.embedding(adata, basis='X_umap_{}'.format(k), s=20, ax = axs[i], show=False, legend_loc=None, **kwargs)
        axs[i].set_title(k)
        axs[i].set_xlabel("UMAP 1")
        axs[i].set_ylabel("UMAP 2")
    fig.tight_layout()
    return fig



# from https://gist.github.com/sumartoyo/edba2eee645457a98fdf046e1b4297e4
def sparse_vars(a, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))