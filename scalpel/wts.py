from scipy.stats import binom_test, entropy
import numpy as np
from fbpca import pca
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

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
