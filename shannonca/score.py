from scipy.stats import binom
from scipy.sparse import csr_matrix, hstack
from scipy.stats import ttest_ind_from_stats, mannwhitneyu
from scipy.stats.distributions import norm
from scipy.stats.mstats import rankdata
from .utils import sparse_vars
from .scorers import *
import numpy as np

import sys
def taylor_exp(x, n):
    # taylor series of 1-(1-x)^n
    # use for multiple test correction if x is very small.
    return (n * x
            - 0.5 * np.power(x, 2) * (n - 1) * n
            + (1. / 6.) * np.power(x, 3) * (n - 2) * (n - 1) * n
            - (1. / 24) * np.power(x, 4) * (n - 3) * (n - 2) * (n - 1) * n)

def log1m_taylor(x):
    return(-1*x -np.power(x,2)/2 - np.power(x,3)/3 - np.power(x,4)/4 -np.power(x,5)/5 - np.power(x,6)/6)

def bootstrapped_ntests(X, trials=1000, k=15, model='binomial', return_all=False, seed=None, **kwargs):
    np.random.seed(seed)
    random_nbhds = np.random.choice(X.shape[0], size=(trials, k), replace=True)

    if issubclass(type(model), Scorer):
        # can directly specify model or give a string to construct one.
        scorer = model
    elif model == 'wilcoxon':
        scorer = WilcoxonScorer()
    elif model == 'binomial':
        scorer = BinomialScorer()
    elif model == 'ttest':
        scorer = TScorer()
    else:  # break
        assert False, 'scorer not found' #TODO fix

    nbhd_scores = scorer.score(X, nbhds=random_nbhds)
    #nbhd_scores = info_score(X, random_nbhds, n_tests=1, model=model, **kwargs)
    nbhd_ps = np.exp(-1 * np.abs(nbhd_scores.todense()))

    nbhd_minps = np.min(nbhd_ps, axis=1).A.flatten()

    sorted_minps = sorted(nbhd_minps)

    # empirical frequencies of low minimum p-values
    xvals = np.array(sorted_minps)
    yvals = np.arange(len(sorted_minps)) / len(sorted_minps)


    log1m_xvals = np.log(1 - xvals)
    log1m_yvals = np.log(1 - yvals)
    log1m_xvals[xvals<1e-10] = log1m_taylor(xvals[xvals<1e-10])

    xvals = log1m_xvals
    yvals = log1m_yvals

    # linear fit
    coef = np.mean(np.multiply(np.array(xvals), np.array(yvals))) / np.mean(np.power(np.array(xvals), 2))
    if return_all:
        return((xvals, yvals, np.ceil(coef).astype('int')))
    else:
        return(np.ceil(coef).astype('int'))

def get_binom_scores(gene_probs, k, max_bins=500, verbose=True, scaled=False, n_tests=50, multi_correct=True,
                     error_rate=1.0):
    # compute binom_test(x, k, p) for all x in 1:k and all p in gene_probs
    # if scaled, put more density in lower probabilities
    if max_bins < float('inf'):
        # split interval into max_bins bins

        if scaled:
            if verbose:
                print('using scaled bins')
            min_prob = np.min(gene_probs[gene_probs > 0])
            precision = (min_prob) ** (1. / float(max_bins))  # fold accuracy in probability approx
            splits = precision ** (max_bins - (np.arange(max_bins)))
        else:
            splits = np.arange(0, 1, 1 / max_bins)[1:]  # don't allow a gene to have 0 prob...

        # bins each gene gets put in, based on its global probability
        gene_bins = np.array([np.argmin(np.abs(gene_probs[j] - splits)) for j in range(len(gene_probs))])

        binom_scores = np.zeros((len(splits), k + 1))

        for i, p in enumerate(splits):
            if verbose:
                print('\r computing binom scores for bin {}/{}'.format(i, len(splits)), end=' ')

            # #print('using fast scores')
            # if fast_scores:
            signs = [1 if p * k <= j else -1 for j in np.arange(k + 1)]
            pmfs = binom.pmf(np.arange(k + 1), k, p)
            orderer = np.argsort(pmfs)

            pvals = np.cumsum(pmfs[orderer])


            if multi_correct:
                # use FWER to correct for testing many genes
                pvals_corrected = 1 - np.power(1 - pvals, n_tests)
                pvals_corrected[pvals < 1e-10] = taylor_exp(pvals[pvals < 1e-10], n_tests)  # more accurate

                pvals = pvals_corrected

            pvals[pvals > error_rate] = 1.  # p<error rate or it didn't happen
            binom_scores[i, orderer] = np.array(signs)[orderer] * -1 * np.log(pvals)

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
            p = gene_probs[i]

            if verbose:
                print('\r computing binom scores for genes {}/{}'.format(i, len(gene_probs)), end=' ')
            if gene_probs[i] == 0 or gene_probs[i] == 1:  # no expression, no score
                continue

            signs = [1 if p * k <= j else -1 for j in np.arange(k + 1)]
            pmfs = binom.pmf(np.arange(k + 1), k, p)
            orderer = np.argsort(pmfs)

            pvals = np.cumsum(pmfs[orderer])

            if multi_correct:
                # use FWER to correct for testing many genes
                pvals_corrected = 1 - np.power(1 - pvals, n_tests)
                pvals_corrected[pvals < 1e-10] = taylor_exp(pvals[pvals < 1e-10], n_tests)  # more accurate

                pvals = pvals_corrected
            # pvals = 1-np.power(1-pvals, n_tests)

            pvals[pvals > error_rate] = 1.  # p<error rate or it didn't happen

            binom_scores[i, orderer] = np.array(signs)[orderer] * -1 * np.log(pvals)
            #binom_scores[i, orderer] = pvals #DEBUG ONLY TODO
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

    #return (gene_bins, binom_scores)


def info_score(X, nbhds, max_bins=float('inf'),
               entropy_normalize=False, fast_version=True, binom_scores=None, gene_bins=None,
               return_bin_info=False, verbose=True, n_tests='auto', model='wilcoxon',
               chunk_size=1000, **kwargs):
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



    if type(nbhds) is np.ndarray:
        nbhds = list(nbhds)

    k = len(nbhds[0])  # how many neighbors?

    if n_tests == 'auto':
        # determine by boostrapping
        n_tests = bootstrapped_ntests(X, k=k, model=model)

    wts = np.zeros((len(nbhds), X.shape[1]))  # too big for large data
    # nbhd_counts = np.zeros(X.shape)  # ditto
    # nbhd_sizes = [len(x) for x in NNs]

    # first compute frequencies of all genes:
    gene_probs = np.array((X > 0).sum(axis=0) / float(X.shape[0])).flatten()


    # frequencies of genes within neighborhoods
    nbhd_probs = np.zeros(X.shape)

    if model == 'ttest':
        data = np.ones(np.sum([len(x) for x in nbhds]))
        col_ind = [item for sublist in nbhds for item in sublist]
        row_ind = [i for i, sublist in enumerate(nbhds) for item in sublist]

        # sparse adjacency matrix of NN graph
        nn_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(nbhds), X.shape[0]))

        # get mean gene expressions within each neighborhood; this matrix may be less sparse
        mean_nbhd_exprs = (nn_matrix * X).astype('int').multiply(1/nn_matrix.sum(axis=1)).tocsr()

        vars = np.zeros((len(nbhds), X.shape[1]))
        for i in range(len(nbhds)): # gotta go cell by cell
            nbrs = np.array(nbhds[i]).flatten()
            gene_diffs = np.power((X[nbrs,:].todense()-mean_nbhd_exprs[i,:].todense()),2) # diffs of gene expression
            vars[i,:] = gene_diffs.mean(axis=0)
        vars = csr_matrix(vars)

        global_means = np.tile(X.mean(axis=0), (len(nbhds),1))

        #sign is pos if mean is higher, negative otherwise.
        signs = 2*(mean_nbhd_exprs.todense() >= global_means).astype('int') - 1

        global_var = np.tile(np.var(X.todense(), axis=0), (len(nbhds),1))
        nobs_global = np.tile(X.shape[0], (len(nbhds), X.shape[1]))
        nobs_local = np.tile(k, (len(nbhds), X.shape[1]))

        wts = ttest_ind_from_stats(mean1=mean_nbhd_exprs.todense().flatten(),
                                              std1=np.array(np.sqrt(vars.todense()).flatten()),
                                              nobs1=np.array(nobs_local).flatten(),
                                              mean2=np.array(global_means).flatten(),
                                              std2=np.array(np.sqrt(global_var)).flatten(),
                                              nobs2=np.array(nobs_global).flatten()).pvalue.reshape((len(nbhds), X.shape[1]))

        np.nan_to_num(wts, copy=False, nan=1.0) # nans become pval 1

        wts[wts==0] = sys.float_info.min # remove zeros


        if n_tests>1:
            # use FWER to correct for testing many genes
            wts_corrected = 1 - np.power(1 - wts, n_tests)
            wts_corrected[wts < 1e-10] = taylor_exp(wts[wts < 1e-10], n_tests)  # more accurate

            wts = wts_corrected
        else:
            wts_corrected = wts

        wts = -1*np.log(wts) # convert to info

        np.nan_to_num(wts, copy=False, nan=1.0)  # nans become pval 1

        wts = np.multiply(signs, wts) # negative if underexpressed





        return(csr_matrix(wts))

    #TODO TODO add signs

    elif model == 'wilcoxon':
        from scipy.stats import rankdata

        def fastRank(array):
            temp = array.argsort(axis=0)
            ranks = np.zeros(temp.shape)

            rows = temp.transpose().flatten()
            cols = np.repeat(np.arange(temp.shape[1]), temp.shape[0])
            ranks[rows, cols] = np.array(list(np.arange(temp.shape[0])) * temp.shape[1])
            return (ranks)

        # Wilcoxon rank sum testa
        #overall_exprs = X.todense().transpose().tolist()

        n_genes = X.shape[1]
        chunk_ends = [0] + list(np.arange(chunk_size, n_genes, chunk_size))
        chunk_ends.append(n_genes)
        gene_idxs = np.array(list(range(n_genes)))
        gene_chunks = [np.array(gene_idxs[chunk_ends[i]:chunk_ends[i + 1]]) for i in range(len(chunk_ends) - 1)]

        wt_blocks = [] # list of sparse matrices to concatenate horizontally

        # make nbhd adjacency matrix
        data = np.ones(np.sum([len(x) for x in nbhds]))
        col_ind = [item for sublist in nbhds for item in sublist]
        row_ind = [i for i, sublist in enumerate(nbhds) for item in sublist]

        # sparse adjacency matrix of NN graph
        nn_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(nbhds), X.shape[0]))

        for i,chunk in enumerate(gene_chunks):
            if verbose:
                print('chunk {}/{}'.format(i+1,len(gene_chunks)), end='\r')

            X_chunk = X[:,chunk]

            wts = rankdata(X_chunk.todense(), axis=0) # gene rankings

            wts = nn_matrix @ wts  # nbhd_ranksums; only want to store one big matrix

            n1 = k
            n2 = X_chunk.shape[0] - k
            sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
            meanrank = n1 * n2 / 2.0

            # #sign is pos if mean rank is higher than average, negative otherwise.
            # signs = 2*(wts >= meanrank).astype('int') - 1

            wts = wts - ((n1 * (n1 + 1)) / 2.0)  # calc U for x, u1

            is_neg = (wts<meanrank) # remember where it was negative

            wts = np.maximum(wts, n1 * n2 - wts)  # bigu

            wts = ((wts - meanrank) / sd) # z values

            wts = 2 * norm.sf(np.abs(wts)) #p values
            #
            # for i in range(len(nbhds)):
            #     print('cell {}/{}'.format(i+1,len(nbhds)+1), end='\r')
            #     #gene_exprs = X[nbhds[i],:].todense()
            #     #all_exprs = np.vstack((gene_exprs, X.todense()))
            #
            #     nbhd_ranks = gene_rankings[nbhds[i],:]
            #
            #     ranksums = np.sum(nbhd_ranks, axis=0)
            #     n1 = k
            #     n2 = X.shape[0]-k
            #     #
            #     # ranks = fastRank(all_exprs.A)
            #     #
            #     # n1 = k
            #     # n2 = X.shape[0]
            #     # ranksums = np.sum(ranks[:k, :], axis=0)
            #     u1 = ranksums - ((n1 * (n1 + 1)) / 2.0)  # calc U for x
            #     u2 = n1 * n2 - u1  # remainder is U for y
            #
            #     sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
            #     meanrank = n1 * n2 / 2.0
            #
            #     bigu = np.maximum(u1, u2)
            #    # wts[i,:] = bigu
            #
            #     z = ((bigu - meanrank) / sd).flatten()
            #     p = 2 * norm.sf(abs(z))
            #     wts[i,:] = p
            #
            #

                # gene_exprs = X[nbhds[i],:].todense().transpose().tolist() # list of gene expression vectors for each nbr
                # for j,local_expr in enumerate(gene_exprs):
                #     global_expr = overall_exprs[j]
                #     wts[i,j] = mannwhitneyu(x=local_expr, y=global_expr, alternative='two-sided', use_continuity=False)[1]
            if n_tests>1:
                # use FWER to correct for testing many genes
                wts[wts> 1e-10] = 1-np.power(1-wts[wts>1e-10], n_tests)
                wts[wts<=1e-10] = taylor_exp(wts[wts <= 1e-10], n_tests)
                # wts_corrected = 1 - np.power(1 - wts, n_tests)
                # wts_corrected[wts < 1e-10] = taylor_exp(wts[wts < 1e-10], n_tests)  # more accurate
                #
                # wts = wts_corrected

            wts = -1*np.log(wts) # convert to info scores

            #sign them
            wts[is_neg] *= -1
            #wts = np.multiply(signs, wts)

            wt_blocks.append(csr_matrix(wts))

        return(hstack(wt_blocks))

    elif model == 'log_likelihood':
        means = X.mean(axis=0)
        variances = sparse_vars(X, axis=0)/float(k)

        data = np.ones(np.sum([len(x) for x in nbhds]))
        col_ind = [item for sublist in nbhds for item in sublist]
        row_ind = [i for i, sublist in enumerate(nbhds) for item in sublist]

        # sparse adjacency matrix of NN graph
        nn_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(nbhds), X.shape[0]))

        nbhd_means = ((nn_matrix * X)/float(k)).todense()

        print(np.min(variances))
        wts = 1 / 2. * np.log(2 * np.pi) + np.power((nbhd_means - means), 2) / (2 * variances) + np.log(variances)/2.

        signs = 2*(nbhd_means >= means).astype('int') - 1

        wts = np.multiply(wts, signs)

        return(csr_matrix(wts))



    elif model == 'binomial':
        X = csr_matrix((X > 0).astype('float'))  # convert to sparse binarized matrix
        if binom_scores is None or gene_bins is None:
            if n_tests is None:
                n_tests = X.shape[1]  # multi-correct per cell
            gene_bins, binom_scores = get_binom_scores(gene_probs, k, max_bins=max_bins,
                                                       verbose=verbose, n_tests=n_tests, **kwargs)

        if fast_version:
            # compute significance of gene expression in each cell's neighborhood
            # first convert neighborhood to sparse matrix
            data = np.ones(np.sum([len(x) for x in nbhds]))
            col_ind = [item for sublist in nbhds for item in sublist]
            row_ind = [i for i, sublist in enumerate(nbhds) for item in sublist]

            # sparse adjacency matrix of NN graph
            nn_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(nbhds), X.shape[0]))

            # get gene expressions within each neighborhood; this matrix may be less sparse
            nbhd_exprs = (nn_matrix * X).astype('int').todense()

            # # extract locations and values of nonzero nbhd expressions.
            # rows, cols = nbhd_exprs.nonzero()
            # exprs = nbhd_exprs.data

            # apply binomial scores
            rows, cols = np.indices((len(nbhds), X.shape[1]))
            rows = rows.flatten()
            cols = cols.flatten()

            wts = binom_scores[gene_bins[cols], np.array(nbhd_exprs[rows, cols]).flatten()].reshape((len(nbhds), X.shape[1]))

        else:
            for i in range(len(nbhds)):
                if verbose:
                    if i < len(nbhds) - 1:
                        print('\r computing counts for cell {}/{}'.format(i, X.shape[0]), end='         ')
                    else:
                        print('\r computing counts for cell {}/{}'.format(i, X.shape[0]), end='         \n')

                nnbhd = X[nbhds[i], :]

                nbhd_size = len(nbhds[i])
                nbhd_gene_counts = np.array((nnbhd > 0).sum(axis=0)).flatten()

                nbhd_probs[i, :] = nbhd_gene_counts / nbhd_size

                if max_bins < float('inf'):
                    # look up the binomial score in the nearest bins

                    # gene_scores = [binom_scores[gene_bins[j], count] for j, count in enumerate(nbhd_gene_counts)]
                    gene_scores = binom_scores[gene_bins, nbhd_gene_counts]

                else:
                    gene_scores = [binom_scores[j, count] for j, count in enumerate(nbhd_gene_counts)]

                wts[i, :] = gene_scores
                # expected_vals = nbhd_size * gene_probs
                # wts[i, :] = -1*np.log(gene_scores) * (2*(nbhd_gene_counts > expected_vals)-1)

    if entropy_normalize:  # divide each column by the entropy of the corresponding gene
        gene_entropies = -1 * (np.multiply(gene_probs, np.log(gene_probs))
                               + np.multiply((1 - gene_probs), np.log(1 - gene_probs)))
        gene_entropies[np.logical_not(np.isfinite(gene_entropies))] = float(
            'inf')  # zeros out non-expressed or everywhere-expressed genes
        wts = np.divide(wts, gene_entropies)

    # if return_all:
    #     return (wts, gene_probs, nbhd_probs)
    wts = csr_matrix(wts)
    if model == 'binomial' and return_bin_info:  # for iteration
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
