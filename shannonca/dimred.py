### stuff that adds dimensionality reductions when called on a scanpy object
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from .score import info_score
import numpy as np
from scipy.sparse import csr_matrix
import warnings

def reduce(X, n_comps = 50, iters=1, max_bins=float('inf'), fast_version=True, scaled_bins=False, nbhds=None,
           rep=None, nbhd_size=15, n_pcs=50, metric='cosine',
           keep_scores=False, keep_loadings=False, keep_all_iters=False, verbose=False, **kwargs):


    """
    :param X: (num cells)x(num_genes)-sized array or sparse matrix to be dimensionality-reduced. Should be nonnegative,
    with 0 indicating no recorded transcripts (required for binarization and binomial inference).
    :param n_comps: Desired dimensionality of the reduction.
    :param iters: Number of iterations. More iterations yi
    :param max_bins: Resolution of gene frequency measurements. If max_bins=inf (default), each global gene frequency is
    measured exactly. Otherwise, they are binned into max_bins bins, reducing the number of binomial tests required.
    :param fast_version: If true (default), runtime will be faster but memory usage will be higher. If false, will be slower with
    a smaller memory footprint.
    :param scaled_bins: If true, frequency bins will be unevenly spaced over the unit interval via a power law,
    ensuring accuracy to a constant factor. If false (default), bins evenly divide the unit interval.
    :param nbhd_size: Size of neighborhoods used to assess the local expression of a gene. Should be smaller than the
    smallest subpopulation; default is 15. Does not drastically affect the
    :param nbhds: Optional - if k-neighborhoods of points are already determined, they can be specified here as
    a (num_cells)*k array. Otherwise, they will be computed from the PCS.
    :param rep: Optional - existing featurization for determining k-neighborhoods of points. If not provided,
    k-neighborhoods will be computed in PCA space.
    :param n_pcs: If either rep or nbhds is provided, this parameter is ignored. Otherwise, the number of PCs used for computing
    the starting representation, from which the starting k-neighborhoods are derived.
    :param metric: if either rep or nbhds is provided, this parameter is ignored. Otherwise, the metric used in computing
    the initial k-neighborhoods
    :param keep_scores: if True, returns information scores for each gene/cell combo in a sparse matrix.
    :param keep_loadings: If True, returns loadings of each gene in each metagene as a dense matrix.
    :param verbose: If True, print progress.
    :param kwargs: Other arguments to be passed to info_score
    :return: If return_scores or return_loadings are both false, a (n cells)x(n_comps)-dimensional array
    of reduced features. Otherwise, a dictionary with keys 'reduction', 'scores' and/or 'loadings'.
    """

    result = {} # final result, if return_all_iters=True

    if nbhds is None:
        if rep is None:
            rep = sc.tl.pca(X, n_comps=n_pcs)

        nn = NearestNeighbors(n_neighbors=nbhd_size, metric=metric)
        nn.fit(rep)

        nbhds = nn.kneighbors(return_distance=False)

        # add points to their own neighborhoods
        nbhds = np.concatenate((np.array(range(X.shape[0])).reshape(-1, 1), nbhds), axis=1)


    for i in range(iters):
        if verbose:
            print('\niteration {}'.format(i+1))

        if i==0:
            #keep binomial scores for future runs
            infos, gene_bins, binom_scores = info_score(X, nbhds,
                                                        max_bins=max_bins,
                                                        return_bin_info=True,
                                                        fast_version=fast_version,
                                                        verbose=verbose,
                                                        scaled=scaled_bins,
                                                        **kwargs)
        else:
            infos = info_score(X, nbhds,
                               max_bins=max_bins,
                               binom_scores=binom_scores,
                               gene_bins=gene_bins,
                               fast_version=fast_version,
                               verbose=verbose,
                               scaled=scaled_bins,
                               **kwargs)


        a, bt, c, d = sc.tl.pca(infos, n_comps=n_comps, return_info=True)

            # pca(infos, k=n_comps, raw=True, pc_iters=pc_iters)

        current_dimred = X @ bt.transpose() # metagene expression values in X

        if keep_all_iters:
            result['reduction_{}'.format(i+1)] = current_dimred

        # compute neighborhoods for next iteration, using current reduction
        if i < (iters-1):
            nn = NearestNeighbors(n_neighbors=nbhd_size, metric=metric)
            nn.fit(current_dimred)

            nbhds = nn.kneighbors(return_distance=False)
            nbhds = np.concatenate((np.array(range(X.shape[0])).reshape(-1, 1), nbhds), axis=1)


    if not keep_scores and not keep_loadings and not keep_all_iters:
        return current_dimred


    if not keep_all_iters:
        result['reduction'] =  current_dimred #store result
    if keep_scores:
        result['scores'] = csr_matrix(infos)
    if keep_loadings:
        result['loadings'] = bt.transpose()

    return result


def reduce_scanpy(adata, keep_scores=False, keep_loadings=True, keep_all_iters=False, layer=None, key_added='sca', iters=1, **kwargs):
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    dimred_info = reduce(X, keep_scores=keep_scores, keep_loadings=keep_loadings,
                           keep_all_iters=keep_all_iters, iters=iters, **kwargs)

    if not keep_scores and not keep_loadings and not keep_all_iters:
        adata.obsm[key_added] = dimred_info
    else:
        if keep_all_iters:
            for i in range(iters):
                adata.obsm[key_added+'_'+str(i+1)] = dimred_info['reduction_{}'.format(i+1)]
        else:
            adata.obsm[key_added] = dimred_info['reduction']
            
        if 'scores' in dimred_info:
            adata.layers[key_added+'_score'] = dimred_info['scores']
        if 'loadings' in dimred_info:
            adata.varm[key_added+'_loadings'] = dimred_info['loadings']


###### DEPRECATED #########

def info_pca(nbhd_size=15, n_neighbors=15, n_init_pcs=50, n_info_pcs=50, key_added='info_pca', metric='cosine',
             max_bins=float('inf'), iters=1, entropy_normalize=False, p_val=True,
             keep_scores=True, binarize=False,  verbose=False, **kwargs):

    warnings.warn('This function has been replaced by reduce and reduce_scanpy, and will soon be removed.' )

    def f(d):
        #sc.tl.pca(d, n_comps=n_init_pcs)

        if binarize:
            pcs = sc.tl.pca((d.X>0).astype('float'), n_comps = n_init_pcs)
            d.obsm['X_pca'] = pcs
            # U,s,Vt = pca((d.X>0).astype('float'), k=n_init_pcs, n_iter=pc_iters)
            # pcs = U*s
        else:
            # U,s,Vt = pca(d.X, k=n_init_pcs, n_iter=5)
            # pcs = U*s
            # d.obsm['X_pca'] = pcs
            sc.tl.pca(d, n_comps=n_init_pcs)
            pcs = d.obsm['X_pca']


        nn = NearestNeighbors(n_neighbors=nbhd_size, metric=metric);
        nn.fit(pcs)
        nbhds = nn.kneighbors(return_distance=False)

        # add points to their own neighborhoods
        nbhds = np.concatenate((np.array(range(d.shape[0])).reshape(-1, 1), nbhds), axis=1)


        for i in range(iters):
            if verbose:
                print('\niteration {}'.format(i))

            if i==0:
                #keep binomial scores for future runs
                infos, gene_bins, binom_scores = info_score((d.X>0).astype('float'), nbhds,
                                                            max_bins=max_bins, entropy_normalize=entropy_normalize,
                                                            return_bin_info=True, **kwargs)
            else:
                infos = info_score((d.X>0).astype('float'), nbhds, max_bins=max_bins,
                                   binom_scores = binom_scores, gene_bins=gene_bins, **kwargs)

            a, bt, c, dt = sc.tl.pca(infos, n_comps=n_info_pcs, return_info=True)
            #U, s, Vt = pca(infos, k=n_info_pcs, raw=True)

            if binarize:
                X = (d.X>0).astype('float').todense()

                d.obsm[key_added] = X @ bt.transpose()
                    #np.matmul(X, Vt.transpose())
            else:
                d.obsm[key_added] = d.X @ bt.transpose()

            d.varm[key_added+'_loadings'] = bt.transpose()

            #print(d.obsm[key_added].shape)

            nn = NearestNeighbors(n_neighbors=nbhd_size);
            nn.fit(d.obsm[key_added])

            dist, nbhds = nn.kneighbors()
            nbhds = np.concatenate((np.array(range(d.shape[0])).reshape(-1, 1), nbhds), axis=1)

        sc.pp.neighbors(d, use_rep=key_added, key_added=key_added, metric=metric, n_neighbors=n_neighbors)

        if keep_scores:
            # info_df = pd.DataFrame(infos)
            # info_df.columns = d.var.index
            # info_df.index = d.obs.index
            d.layers[key_added+'_score'] = csr_matrix(infos)

    return f