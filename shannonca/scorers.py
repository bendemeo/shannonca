from .utils import to_sparse_adjacency
from scipy.stats import rankdata
import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse
from scipy.stats import norm, binom, ttest_ind_from_stats
import sys


class Scorer:
    def __init__(self, corrector=None, seed=None, verbose=False):
        self.corrector = corrector
        self.seed = seed
        self.verbose = verbose

    def score(self, X, nbhds, nn_matrix = None):
        k = len(nbhds[0])
        # if self.n_tests == 'auto':  # compute by bootstrapping
        #     self.n_tests = 1
        #     self.n_tests = bootstrapped_ntests(X, scorer=self, return_all=False, seed=self.seed)
        pass


class Tf_idfScorer(Scorer):

    def score(self, X, nbhds = None):
        #nbhds is ignored; just TF-IDF transform the data.
        if not issparse(X):
            X = csr_matrix(X)

        X_binarized = (X>0).astype("float")
        counts = X_binarized.sum(axis=0).A.flatten()
        idfs = np.ones(len(counts))
        idfs[counts>0] = np.log(X.shape[0]/counts[counts>0])

        scores = X.multiply(idfs)
        return(scores)

class BinomialScorer(Scorer):
    def __init__(self, corrector=None, seed=None, verbose=False, scaled=False,
                 fast_version=True, max_bins=float('inf')):
        super().__init__(corrector, seed, verbose)
        self.fast_version = fast_version
        self.max_bins = max_bins
        self.scaled = scaled
        self.verbose = verbose

    def get_binom_scores(self, gene_probs, k, max_bins=500, verbose=True, scaled=False,
                     error_rate=1.0):
        # compute binom_test(x, k, p) for all x in 1:k and all p in gene_probs
        # if scaled, put more density in lower probabilities
        if max_bins < float('inf'):
            # split interval into max_bins bins
            if self.scaled:
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
                if self.corrector is not None:
                    pvals = self.corrector.correct(pvals)

                pvals[pvals > error_rate] = 1.  # p<error rate or it didn't happen
                binom_scores[i, orderer] = np.array(signs)[orderer] * -1 * np.log(pvals)

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

                if self.corrector is not None:
                    pvals = self.corrector.correct(pvals)
                # pvals = 1-np.power(1-pvals, n_tests)

                pvals[pvals > error_rate] = 1.  # p<error rate or it didn't happen

                binom_scores[i, orderer] = np.array(signs)[orderer] * -1 * np.log(pvals)
        return (gene_bins, binom_scores)

    def score(self, X, nbhds, nn_matrix=None, binom_scores=None, gene_bins=None):
        super().score(X, nbhds)
        k = len(nbhds[0])
        gene_probs = np.array((X > 0).sum(axis=0) / float(X.shape[0])).flatten()
        nbhd_probs = np.zeros(X.shape)
        wts = np.zeros((len(nbhds), X.shape[1]))  # too big for large data

        X = csr_matrix((X > 0).astype('float'))  # convert to sparse binarized matrix
        if binom_scores is None or gene_bins is None:
            gene_bins, binom_scores = self.get_binom_scores(gene_probs, k)
        if self.fast_version:
            # compute significance of gene expression in each cell's neighborhood
            # first convert neighborhood to sparse matrix

            if nn_matrix is None:
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

            wts = binom_scores[gene_bins[cols], np.array(nbhd_exprs[rows, cols]).flatten()].reshape(
                (len(nbhds), X.shape[1]))

        else:
            for i in range(len(nbhds)):
                if self.verbose:
                    if i < len(nbhds) - 1:
                        print('\r computing counts for cell {}/{}'.format(i, X.shape[0]), end='         ')
                    else:
                        print('\r computing counts for cell {}/{}'.format(i, X.shape[0]), end='         \n')

                nnbhd = X[nbhds[i], :]

                nbhd_size = len(nbhds[i])
                nbhd_gene_counts = np.array((nnbhd > 0).sum(axis=0)).flatten()

                nbhd_probs[i, :] = nbhd_gene_counts / nbhd_size

                if self.max_bins < float('inf'):
                    # look up the binomial score in the nearest bins

                    # gene_scores = [binom_scores[gene_bins[j], count] for j, count in enumerate(nbhd_gene_counts)]
                    gene_scores = binom_scores[gene_bins, nbhd_gene_counts]

                else:
                    gene_scores = [binom_scores[j, count] for j, count in enumerate(nbhd_gene_counts)]

                wts[i, :] = gene_scores
                # expected_vals = nbhd_size * gene_probs
                # wts[i, :] = -1*np.log(gene_scores) * (2*(nbhd_gene_counts > expected_vals)-1)
        return(csr_matrix(wts))

class TScorer(Scorer):
    def __init__(self, corrector=None, seed=None, verbose=False):
        super().__init__(corrector, seed, verbose)
        self.verbose =  verbose
        self.seed = seed
        self.corrector = corrector

    def score(self, X, nbhds, nn_matrix = None):
        if not issparse(X):
            X = csr_matrix(X)

        k = len(nbhds[0])
        if nn_matrix is None:
            data = np.ones(np.sum([len(x) for x in nbhds]))
            col_ind = [item for sublist in nbhds for item in sublist]
            row_ind = [i for i, sublist in enumerate(nbhds) for item in sublist]

            # sparse adjacency matrix of NN graph
            nn_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(nbhds), X.shape[0]))

        # get mean gene expressions within each neighborhood; this matrix may be less sparse
        mean_nbhd_exprs = (nn_matrix * X).astype('int').multiply(1 / nn_matrix.sum(axis=1)).tocsr()

        vars = np.zeros((len(nbhds), X.shape[1]))
        for i in range(len(nbhds)):  # gotta go cell by cell
            nbrs = np.array(nbhds[i]).flatten()
            gene_diffs = np.power((X[nbrs, :].todense() - mean_nbhd_exprs[i, :].todense()),
                                  2)  # diffs of gene expression
            vars[i, :] = gene_diffs.mean(axis=0)
        vars = csr_matrix(vars)

        global_means = np.tile(X.mean(axis=0), (len(nbhds), 1))

        # sign is pos if mean is higher, negative otherwise.
        signs = 2 * (mean_nbhd_exprs.todense() >= global_means).astype('int') - 1

        global_var = np.tile(np.var(X.todense(), axis=0), (len(nbhds), 1))
        nobs_global = np.tile(X.shape[0], (len(nbhds), X.shape[1]))
        nobs_local = np.tile(k, (len(nbhds), X.shape[1]))

        wts = ttest_ind_from_stats(mean1=mean_nbhd_exprs.todense().flatten(),
                                   std1=np.array(np.sqrt(vars.todense()).flatten()),
                                   nobs1=np.array(nobs_local).flatten(),
                                   mean2=np.array(global_means).flatten(),
                                   std2=np.array(np.sqrt(global_var)).flatten(),
                                   nobs2=np.array(nobs_global).flatten()).pvalue.reshape((len(nbhds), X.shape[1]))

        np.nan_to_num(wts, copy=False, nan=1.0)  # nans become pval 1

        wts[wts == 0] = sys.float_info.min  # remove zeros

        if self.corrector is not None:
            wts = self.corrector.correct(wts)

        wts = -1 * np.log(wts)  # convert to info

        np.nan_to_num(wts, copy=False, nan=1.0)  # nans become pval 1

        wts = np.multiply(signs, wts)  # negative if underexpressed

        return (csr_matrix(wts))

class WilcoxonScorer(Scorer):
    def __init__(self, corrector = None, seed=None, verbose=False):
        super().__init__(corrector, seed, verbose)

    def score(self, X, nbhds, nn_matrix=None):
        if not issparse(X):
            X = csr_matrix(X)

        k = len(nbhds[0])

        super().score(X, nbhds)  # handle multi-test factor determination
        # Wilcoxon rank sum testa
        # overall_exprs = X.todense().transpose().tolist()

        n_genes = X.shape[1]

        if nn_matrix is None:
            nn_matrix = to_sparse_adjacency(nbhds, n_cells=X.shape[0])

        wts = rankdata(X.todense(), axis=0)  # gene rankings
        wts = nn_matrix @ wts  # nbhd_ranksums; only want to store one big matrix

        n1 = k
        n2 = X.shape[0] - k
        sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        meanrank = n1 * n2 / 2.0

        # #sign is pos if mean rank is higher than average, negative otherwise.
        # signs = 2*(wts >= meanrank).astype('int') - 1

        wts = wts - ((n1 * (n1 + 1)) / 2.0)  # calc U for x, u1

        is_neg = (wts < meanrank)  # remember where it was negative

        wts = np.maximum(wts, n1 * n2 - wts)  # bigu

        wts = ((wts - meanrank) / sd)  # z values

        wts = 2 * norm.sf(np.abs(wts))  # p values

        if self.corrector is not None:
            wts = self.corrector.correct(wts)

        wts = -1 * np.log(wts)  # convert to info scores

        # sign them
        wts[is_neg] *= -1

        return(csr_matrix(wts))

class ChunkedScorer(Scorer):

    def __init__(self, base_scorer, chunk_size=1000, max_size=None):

        self.verbose = base_scorer.verbose
        self.base_scorer = base_scorer
        self.chunk_size = chunk_size
        self.max_size = max_size

    def score(self, X, nbhds, **kwargs):

        #amortize this over all chunks.
        nn_matrix = to_sparse_adjacency(nbhds, n_cells=X.shape[0])

        if self.max_size is not None:
            # set chunksize so that at most maxsize floats are stored in a dense matrix at once
            chunksize = np.ceil(float(self.max_size) / X.shape[0]).astype('int')
        else:
            chunksize = self.chunk_size

        n_genes = X.shape[1]
        chunk_ends = [0] + list(np.arange(chunksize, n_genes, chunksize))
        chunk_ends.append(n_genes)
        gene_idxs = np.array(list(range(n_genes)))
        gene_chunks = [np.array(gene_idxs[chunk_ends[i]:chunk_ends[i + 1]]) for i in range(len(chunk_ends) - 1)]

        wt_blocks = [] # list of sparse matrices to concatenate horizontally

        for i,chunk in enumerate(gene_chunks):
            if self.verbose:
                print('chunk {}/{}'.format(i+1,len(gene_chunks)), end='\r')
            X_chunk = X[:,chunk]

            wt_blocks.append(self.base_scorer.score(X_chunk, nbhds=nbhds, nn_matrix=nn_matrix, **kwargs))

        return (csr_matrix(hstack(wt_blocks)))

