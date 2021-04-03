from scipy.stats import binom_test, binom, entropy
import numpy as np
from scipy.sparse import csr_matrix


def taylor_exp(x, n):
    # taylor series of 1-(1-x)^n
    # use for multiple test correction if x is very small.
    return (n*x-
           0.5*np.power(x,2)*(n-1)*n+
           (1./6.)*np.power(x,3) * (n-2)*(n-1)*n-
           (1./24)*np.power(x,4)*(n-3)*(n-2)*(n-1)*n)

def get_binom_scores(gene_probs, k, max_bins=500,  verbose=True, scaled=False, n_tests=50, multi_correct=True, error_rate=1.0):
    # compute binom_test(x, k, p) for all x in 1:k and all p in gene_probs
    # if scaled, put more density in lower probabilities
    if max_bins < float('inf'):
        # split interval into max_bins bins

        if scaled:
            if verbose:
                print('using scaled bins')
            min_prob = np.min(gene_probs[gene_probs>0])
            precision = (min_prob)**(1./float(max_bins)) # fold accuracy in probability approx
            splits = precision ** (max_bins-(np.arange(max_bins)))
        else:
            splits = np.arange(0, 1, 1/max_bins)[1:] # don't allow a gene to have 0 prob...

        #bins each gene gets put in, based on its global probability
        gene_bins = np.array([np.argmin(np.abs(gene_probs[j] - splits)) for j in range(len(gene_probs))])

        binom_scores = np.zeros((len(splits), k+1))

        for i, p in enumerate(splits):
            if verbose:
                print('\r computing binom scores for bin {}/{}'.format(i, len(splits)), end=' ')

            # #print('using fast scores')
            # if fast_scores:
            signs = [1 if p*k <= j else -1 for j in np.arange(k+1)]
            pmfs = binom.pmf(np.arange(k+1), k, p)
            orderer = np.argsort(pmfs)

            pvals = np.cumsum(pmfs[orderer])





            if multi_correct:
                # use FWER to correct for testing many genes
                pvals_corrected = 1-np.power(1-pvals, n_tests)
                pvals_corrected[pvals<1e-10] = taylor_exp(pvals[pvals<1e-10], n_tests) # more accurate

                pvals = pvals_corrected

            pvals[pvals > error_rate] = 1.  # p<error rate or it didn't happen
            binom_scores[i,orderer] = np.array(signs)[orderer] * -1*np.log(pvals)

            # else:
            #     for j in range(k+1):
            #         alt = 'two-sided' if two_tailed else 'less' if float(j)/k < p*k else 'greater'
            #
            #         sign = 1 if p*k <= j else -1  # negative if lower than expected
            #
            #         binom_scores[i,j] = sign * -1*np.log(binom_test(j, n=k, p=p, alternative=alt))


    else:  # compute for all gene probs
        binom_scores = np.zeros((len(gene_probs), k + 1))
        gene_bins = np.array(range(len(gene_probs)))

        for i in range(len(gene_probs)):
            p=gene_probs[i]

            if verbose:
                print('\r computing binom scores for genes {}/{}'.format(i, len(gene_probs)), end=' ')
            if gene_probs[i] == 0 or gene_probs[i] == 1: # no expression, no score
                continue

            signs = [1 if p*k <= j else -1 for j in np.arange(k+1)]
            pmfs = binom.pmf(np.arange(k+1), k, p)
            orderer = np.argsort(pmfs)

            pvals = np.cumsum(pmfs[orderer])


            if multi_correct:
                # use FWER to correct for testing many genes
                pvals_corrected = 1-np.power(1-pvals, n_tests)
                pvals_corrected[pvals<1e-10] = taylor_exp(pvals[pvals<1e-10], n_tests) # more accurate

                pvals = pvals_corrected
               #pvals = 1-np.power(1-pvals, n_tests)

            pvals[pvals > error_rate] = 1.  # p<error rate or it didn't happen

            binom_scores[i,orderer] = np.array(signs)[orderer] * -1*np.log(pvals)
            #
            # for j in range(k + 1):
            #
            #
            #
            #     p = gene_probs[i]
            #     alt = 'two-sided' if two_tailed else 'less' if float(j)/k < p*k else 'greater'
            #
            #     sign = 1 if p*k <= j else -1  # negative if lower than expected
            #
            #     binom_scores[i,j] = sign * -1*np.log(binom_test(j, n=k, p=p, alternative=alt))

    return (gene_bins, binom_scores)

def info_score(X, nbhds, max_bins=float('inf'),
               entropy_normalize=False, fast_version=True, binom_scores=None, gene_bins=None,
               return_bin_info=False, verbose=True, n_tests = 50, **kwargs):
    """
    :param X: sparse count matrix
    :param nbhds: list with indices of nearest neighbors for each obs in X, e.g. from kneighbors() in sklearn
    :param max_bins: Resolution at which global gene probabilities are computed.
    if inf, genes get their own probabilities. Otherwise, the unit interval is split into max_bins pieces
    and they are rounded. This makes it faster with little performance difference
    :param return_all: if True, will also return global and local gene probabilities

    :param binom_scores: pass in binomial scores for each gene/bin, if pre-computed. Allows saving for future iterations.
    :param gene_bins: pass in gene bins from previous run. Speeds up iteration

    :param return_bin_info: for iteration: keep information about gene bins and binomial probs.
    :param fast_version: if True, use matrix multiplication instead of iteration. Fast, but memory-intensive.
    :return: dense matrix of gene/cell weightings.
    """

    X = csr_matrix((X>0).astype('float'))  # convert to sparse binarized matrix

    if type(nbhds) is np.ndarray:
        nbhds = list(nbhds)

    k = len(nbhds[0]) # how many neighbors?

    wts = np.zeros(X.shape)  # too big for large data
    # nbhd_counts = np.zeros(X.shape)  # ditto
    # nbhd_sizes = [len(x) for x in NNs]

    # first compute frequencies of all genes:
    gene_probs = np.array((X > 0).sum(axis=0) / float(X.shape[0])).flatten()

    #frequencies of genes within neighborhoods
    nbhd_probs = np.zeros(X.shape)


    if binom_scores is None or gene_bins is None:
        if n_tests is None:
            n_tests = X.shape[1] # multi-correct per cell
        gene_bins, binom_scores = get_binom_scores(gene_probs, k, max_bins=max_bins,
                                                   verbose=verbose, n_tests = n_tests, **kwargs)




    if fast_version:
        # compute significance of gene expression in each cell's neighborhood
        #first convert neighborhood to sparse matrix
        data = np.ones(np.sum([len(x) for x in nbhds]))
        col_ind = [item for sublist in nbhds for item in sublist]
        row_ind = [i for i,sublist in enumerate(nbhds) for item in sublist]

        #sparse adjacency matrix of NN graph
        nn_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(X.shape[0], X.shape[0]))

        # get gene expressions within each neighborhood; this matrix may be less sparse
        nbhd_exprs = (nn_matrix * X).astype('int').todense()

        # # extract locations and values of nonzero nbhd expressions.
        # rows, cols = nbhd_exprs.nonzero()
        # exprs = nbhd_exprs.data

        #apply binomial scores
        rows, cols = np.indices((X.shape))
        rows = rows.flatten()
        cols = cols.flatten()

        wts = binom_scores[gene_bins[cols], np.array(nbhd_exprs[rows, cols]).flatten()].reshape(X.shape)

    else:
        for i in range(X.shape[0]):

            if verbose:
                if i < X.shape[0]-1:
                    print('\r computing counts for cell {}/{}'.format(i, X.shape[0]), end='         ')
                else: print('\r computing counts for cell {}/{}'.format(i, X.shape[0]), end='         \n')

            nnbhd = X[nbhds[i], :]

            nbhd_size = len(nbhds[i])
            nbhd_gene_counts = np.array((nnbhd > 0).sum(axis=0)).flatten()

            nbhd_probs[i,:] = nbhd_gene_counts/nbhd_size

            if max_bins < float('inf'):
                # look up the binomial score in the nearest bins

                # gene_scores = [binom_scores[gene_bins[j], count] for j, count in enumerate(nbhd_gene_counts)]
                gene_scores = binom_scores[gene_bins, nbhd_gene_counts]

            else:
                gene_scores = [binom_scores[j, count] for j, count in enumerate(nbhd_gene_counts)]

            wts[i,:] = gene_scores
            # expected_vals = nbhd_size * gene_probs
            # wts[i, :] = -1*np.log(gene_scores) * (2*(nbhd_gene_counts > expected_vals)-1)





    if entropy_normalize: # divide each column by the entropy of the corresponding gene
        gene_entropies = -1*(np.multiply(gene_probs,np.log(gene_probs))+
                             np.multiply((1-gene_probs),np.log(1-gene_probs)))
        gene_entropies[np.logical_not(np.isfinite(gene_entropies))] = float('inf') # zeros out non-expressed or everywhere-expressed genes
        wts = np.divide(wts, gene_entropies)

    # if return_all:
    #     return (wts, gene_probs, nbhd_probs)

    if return_bin_info: # for iteration
        return (wts, gene_bins, binom_scores)
    else:
        return (wts)





    #
    # # pre-compute binomial scores to avoid doing it for each cell/gene
    # if max_bins < float('inf') and binom_scores is None:
    #     # split interval into max_bins bins
    #     splits = np.arange(0, 1, 1/max_bins)[1:] # don't allow a gene to have 0 prob...
    #     gene_bins = np.array([np.argmin(np.abs(gene_probs[j] - splits)) for j in range(X.shape[1])])
    #
    #

    #     binom_scores = np.zeros((len(splits), k+1))
    #     for i, p in enumerate(splits):
    #         print('\r computing binom scores for bin {}/{}'.format(i, len(splits)), end=' ')
    #         for j in range(k+1):
    #             alt = 'two-sided' if two_tailed else 'less' if float(j)/k < p*k else 'greater'
    #
    #             if p_val: # information of any occurrence more extreme
    #                 binom_scores[i,j] = binom_test(j, n=k, p=p, alternative=alt)
    #             else: #information of this actual occurrence
    #                 val = -1*(j*np.log(p)+(k-j)*np.log(1-p))
    #                 if np.isfinite(val): # zero out nans
    #                     binom_scores[i,j] = -1*(j*np.log(p)+(k-j)*np.log(1-p))

    #
    # elif binom_scores is None: # compute for all gene probs
    #     binom_scores = np.zeros((X.shape[1], k + 1))
    #     for i in range(X.shape[1]):
    #         print('\r computing binom scores for genes {}/{}'.format(i, X.shape[1]), end=' ')
    #         for j in range(k + 1):
    #             p=gene_probs[i]
    #             alt = 'two-sided' if two_tailed else 'less' if float(j)/k < p*k else 'greater'
    #             if p_val: # information of any occurrence more extreme
    #                 binom_scores[i,j] = binom_test(j, n=k, p=p, alternative=alt)
    #             else: #information of this actual occurrence
    #                 val = -1*(j*np.log(p)+(k-j)*np.log(1-p))
    #                 if np.isfinite(val): # zero out nans
    #                     binom_scores[i,j] = -1*(j*np.log(p)+(k-j)*np.log(1-p))
