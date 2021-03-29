"""a bunch of metrics"""
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.sparse import issparse

def diff_corr(X, precompute=True):
    """X is your gene-specific vectors"""

    if precompute:
        print('precomputing corrs...')
        gene_scores = np.abs(np.corrcoef(X.todense().transpose())**2)
        print('done')


    def d(x,y): # x and y should be arrays
        sym_diff = ((x>0)!=(y>0))

        if issparse(sym_diff):
            sym_diff = sym_diff.todense()

        if len(sym_diff.shape) > 1:
            sym_diff = np.array(sym_diff).flatten()

        if np.sum(sym_diff) == 0: # all genes the same!
            return 0


        if precompute:
            pairwise_scores = gene_scores[sym_diff,:][:,sym_diff]

        else:
            diff_exprs = (X[:, sym_diff] > 0)  # expressions of genes not in diff
            # print(diff_exprs.todense())
            if issparse(diff_exprs):
                pairwise_scores = np.abs(np.corrcoef(diff_exprs.todense().transpose())**2)
            else:
                pairwise_scores = np.abs(np.corrcoef(np.transpose(diff_exprs))**2)
        #pairwise_jaccards = pairwise_distances(diff_exprs.transpose(), metric='jaccard')

        #print(pairwise_corrs)
        return np.sqrt(np.sum(pairwise_scores))


    return d

