"""ways of weighting cells for gene significance"""
from scipy.stats import binom_test, entropy
import numpy as np
from fbpca import pca
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

"""place for functions that take a count matrix and output a weighting for each gene and cell"""

def info_score(X, NNs, weighted_dir=True, return_all=False, max_bins=float('inf')):
    """
    :param X: sparse count matrix, binarized
    :param NNs: array with indices of nearest neighbors for each obs in X, e.g. from kneighbors() in sklearn
    :param weighted_dir: Whether to compute signed weights based on gene over/underexpression. Recommended.
    :param return_all: if True, will also return global and local gene probabilities
    :param max_bins: Resolution at which global gene probabilities are computed.
    if inf, genes get their own probabilities. Otherwise, the unit interval is split into max_bins pieces
    and they are rounded. This makes it faster with little performance difference
    :return: dense matrix of gene/cell weightings.
    """


    if type(NNs) is np.ndarray:
        k = NNs.shape[1]  # number of neighbors
        NNs = list(NNs)

    wts = np.zeros(X.shape)  # too big for large data
    nbhd_counts = np.zeros(X.shape)  # ditto
    nbhd_sizes = [len(x) for x in NNs]


    # first compute frequencies of all genes:
    gene_probs = np.array((X > 0).sum(axis=0) / float(X.shape[0])).flatten()


    #frequencies of genes within neighborhoods
    nbhd_probs = np.zeros(X.shape)


    # pre-compute binomial scores to avoid doing it for each cell/gene
    if max_bins < float('inf'):
        # split interval into max_bins bins
        splits = np.arange(0, 1, 1/max_bins)[1:] # don't allow a gene to have 0 prob...
        gene_bins = [np.argmin(np.abs(gene_probs[j] - splits)) for j in range(X.shape[1])]


        binom_scores = np.zeros((len(splits), k+1))
        for i, p in enumerate(splits):
            print('\r computing binom scores for bin {}/{}'.format(i, len(splits)), end=' ')
            for j in range(k+1):
                alt = 'less' if float(j)/k < p*k else 'greater'
                binom_scores[i,j] = binom_test(j, n=k, p=p, alternative=alt)


    else: # compute for all gene probs
        binom_scores = np.zeros((X.shape[1], k + 1))
        for i in range(X.shape[1]):
            print('\r computing binom scores for genes {}/{}'.format(i, X.shape[1]), end=' ')
            for j in range(k + 1):
                p=gene_probs[i]
                alt = 'less' if float(j)/k < p*k else 'greater'
                binom_scores[i, j] = binom_test(j, n=k, p=p, alternative=alt)

    # compute significance of gene expression in each cell's neighborhood
    for i in range(X.shape[0]):
        print('\r computing counts for cell {}/{}'.format(i, X.shape[0]), end='         ')
        nnbhd = X[NNs[i], :]

        nbhd_size = len(NNs[i])
        nbhd_gene_counts = np.array((nnbhd > 0).sum(axis=0)).flatten()

        nbhd_probs[i,:] = nbhd_gene_counts/nbhd_size

        if max_bins < float('inf'):
            # look up the binomial score in the nearest bins



            # gene_scores = [binom_scores[gene_bins[j], count] for j, count in enumerate(nbhd_gene_counts)]
            gene_scores = binom_scores[gene_bins, nbhd_gene_counts]

        else:
            gene_scores = [binom_scores[j, count] for j, count in enumerate(nbhd_gene_counts)]

        if weighted_dir:
            expected_vals = nbhd_size * gene_probs
            wts[i, :] = -1*np.log(gene_scores) * (2*(nbhd_gene_counts > expected_vals)-1)

        else:
            wts[i, :] = -1 * np.log(gene_scores)

    if return_all:
        return (wts, gene_probs, nbhd_probs)
    else:
        return (wts)

def gene_info(n_neighbors, n_pcs, norm=None, weighted_dir=False, times_expr = False): # same as above
    def f(X):
        # first build a KNN graph
        print('building knn graph...')
        U, s, Vt = pca(X, k=n_pcs);
        X_pca = U * s

        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X_pca)
        NNs = nn.kneighbors(return_distance=False)

        wts = info_score(X, NNs, weighted_dir=weighted_dir)

        if norm is not None:
            wts = normalize(wts, norm=norm)

        if times_expr:
            return np.multiply(wts, X.todense())
        else:
            return wts

    return f


 ######## Deprecated ##########
# def entropy_score(X, NNs, weighted_dir=False):
#     k = NNs.shape[1]  # number of neighbors
#     NNs = list(NNs)
#
#     wts = np.zeros(X.shape)  # too big for large data
#
#     for i in range(X.shape[0]):
#         print('\r computing entropy for cell {}/{}'.format(i, X.shape[0]), end='         ')
#         #print(np.concatenate(([i], NNs[i])))
#
#         nnbhd = X[np.concatenate(([i], NNs[i])), :]
#
#         nbhd_size = len(NNs[i])
#         nbhd_gene_counts = np.array((nnbhd > 0).sum(axis=0)).flatten()
#
#         nbhd_gene_probs = nbhd_gene_counts/nbhd_size
#
#         log_probs = np.log(nbhd_gene_probs, where = (nbhd_gene_probs>0), out=np.zeros(len(nbhd_gene_probs)))
#         log_inv_probs = np.log(1-nbhd_gene_probs, where = (nbhd_gene_probs<1), out=np.zeros(len(nbhd_gene_probs)))
#
#         gene_entropies =-1*(np.multiply((nbhd_gene_probs), log_probs) + \
#                          np.multiply((1-nbhd_gene_probs),log_inv_probs))
#
#         #gene_entropies = [entropy([x,1-x]) for x in nbhd_gene_probs]
#         wts[i,:] = gene_entropies
#
#
#     return (wts)
#

#
# def binom_logp(X, NNs):
#     """do the neighbors of X  have gene distributions obeying the global?"""
#
#     NNs = list(NNs)
#
#     wts = np.zeros(X.shape) # too big for large data
#
#     #first compute frequencies of all genes:
#     gene_probs = (X>0).sum(axis=0)/float(X.shape[0])\
#
#     for i in range(X.shape[0]):
#         nnbhd = X[NNs[i],:]
#
#         nbhd_size = len(NNs[i])def binom_logp(X, NNs):
#     """do the neighbors of X  have gene distributions obeying the global?"""
#
#     NNs = list(NNs)
#
#     wts = np.zeros(X.shape) # too big for large data
#
#     #first compute frequencies of all genes:
#     gene_probs = (X>0).sum(axis=0)/float(X.shape[0])\
#
#     for i in range(X.shape[0]):
#         nnbhd = X[NNs[i],:]
#
#         nbhd_size = len(NNs[i])
#         nbhd_gene_counts = (nnbhd>0).sum(axis=0)
#         gene_scores = [binom_test(x, n=nbhd_size, p=gene_probs[j]) for j,x in enumerate(nbhd_gene_counts)]
#         wts[i,:] = gene_scores
#
#     return wts
#         nbhd_gene_counts = (nnbhd>0).sum(axis=0)
#         gene_scores = [binom_test(x, n=nbhd_size, p=gene_probs[j]) for j,x in enumerate(nbhd_gene_counts)]
#         wts[i,:] = gene_scores
#
#     return wts

