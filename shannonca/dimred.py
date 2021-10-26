import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from .score import info_score, bootstrapped_ntests
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
from .scorers import Scorer, WilcoxonScorer, BinomialScorer, ChunkedScorer, TScorer
from .correctors import FWERCorrector
from .embedders import SCAEmbedder
from .connectors import MetricConnector, Connector

def reduce(X, n_comps=50, iters=1, nbhds=None,
           nbhd_size=15,  metric='euclidean', model='wilcoxon',
           keep_scores=False, keep_loadings=False, keep_all_iters=False, verbose=False, n_tests = 'auto',
           seed=10,  chunk_size=None, **kwargs):
    """Compute an SCA reduction of the input data

    :param X: (num cells)x(num_genes)-sized array or sparse matrix to be dimensionality-reduced.
    :type X: numpy.ndarray | scipy.spmatrix
    :param n_comps: Desired dimensionality of the reduction. Default 50
    :type n_comps: int
    :param iters: Number of iterations of SCA. More iterations usually strengthens signal, stabilizing around 3-5
    :type iters: int
    :param nbhd_size: Size of neighborhoods used to assess the local expression of a gene. Should be smaller than the smallest subpopulation; default is 15.
    :type nbhd_size: int
    :param model: Model used to test for local enrichment of genes, used to compute information scores. One of ["wilcoxon","binomial","ttest"], default "wilcoxon" (recommended).
    :type model: str
    :param nbhds: Optional - if k-neighborhoods of points are already determined, they can be specified here as a (num_cells)*k array or list. Otherwise, they will be computed from the PCA embedding. Default None
    :type nbhds: numpy.ndarray | list
    :param metric: Metric used to compute k-nearest neighbor graphs for SCA score computation. Default "euclidean". See sklearn.neighbors.DistanceMetric for list of choices.
    :type metric: str
    :param keep_scores: if True, keep and return the information score matrix. Default False.
    :type keep_scores: bool
    :param keep_loadings: If True, returns loadings of each gene in each metagene as a dense matrix. Default False.
    :type keep_loadings: bool
    :param verbose: If True, print progress. Default False
    :type verbose: bool
    :param n_tests: Effective number of independent genes per cell, use for FWER multiple testing correction. Set to "auto" to automatically determine by bootstrapping. Default "auto".
    :type n_tests: str | int
    :param kwargs: Other arguments to be passed to the chosen scorer
    :return: If return_scores or return_loadings are both false, a (n cells)x(n_comps)-dimensional array of reduced features. Otherwise, a dictionary with keys 'reduction', 'scores' and/or 'loadings'.
    :rtype: numpy.ndarray | dict

    """

    if n_tests == 'auto':
        n_tests = bootstrapped_ntests(X, model=model, k=nbhd_size, seed=seed)
        if verbose:
            print('multi-testing correction for {} features'.format(n_tests))

    corrector = FWERCorrector(n_tests=n_tests)

    if issubclass(type(model), Scorer):
        # can directly specify model or give a string to construct one.
        scorer = model
    elif model == 'wilcoxon':
        scorer = WilcoxonScorer(corrector=corrector, verbose=verbose, **kwargs)
    elif model == 'binomial':
        scorer = BinomialScorer(corrector=corrector, verbose=verbose, **kwargs)
    elif model == 'ttest':
        scorer = TScorer(corrector=corrector, verbose=verbose, **kwargs)
    else:  # break
        assert False, 'scorer not found' #TODO fix

    if chunk_size is not None:
        scorer = ChunkedScorer(base_scorer=scorer, chunk_size=chunk_size)

    if issubclass(type(metric), Connector):
        connector = metric
    else:
        connector = MetricConnector(n_neighbors=nbhd_size, metric=metric, include_self=True)

    embedder = SCAEmbedder(scorer=scorer, connector=connector, n_comps=n_comps, iters=iters)
    dimred = embedder.embed(X, keep_scores=keep_scores, keep_loadings=keep_loadings, keep_all_iters=keep_all_iters,
                            nbhds=nbhds)

    if not keep_scores and not keep_loadings and not keep_all_iters:
        return dimred

    result = {}
    if not keep_all_iters:
        result['reduction'] = dimred  # store result
    else:
        result.update({('reduction_'+str(i)):j for i,j in embedder.embedding_dict.items()})
    if keep_scores:
        result['scores'] = embedder.scores
    if keep_loadings:
        result['loadings'] = embedder.loadings

    return result

def reduce_scanpy(adata, keep_scores=False, keep_loadings=True, keep_all_iters=False, layer=None, key_added='sca',
                  iters=1, model='wilcoxon',**kwargs):
    """
    Compute an SCA reduction of the given dataset, stored as a scanpy AnnData.

    :param adata: AnnData object containing single-cell transcriptomic data to be reduced
    :type adata: scanpy.AnnData
    :param keep_scores: if True, stores information score matrix in adata.layers[key_added+'_score']. Default False.
    :type keep_scores: bool

    """

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X


    dimred_info = reduce(X, keep_scores=keep_scores, keep_loadings=keep_loadings,
                         keep_all_iters=keep_all_iters, iters=iters, model=model, **kwargs)

    if not keep_scores and not keep_loadings and not keep_all_iters:
        adata.obsm['X_'+key_added] = dimred_info
    else:
        if keep_all_iters:
            for i in range(iters):
                adata.obsm['X_'+key_added + '_' + str(i + 1)] = dimred_info['reduction_{}'.format(i + 1)]
        else:
            adata.obsm['X_'+key_added] = dimred_info['reduction']

        if 'scores' in dimred_info:
            adata.layers[key_added + '_score'] = dimred_info['scores']
        if 'loadings' in dimred_info:
            adata.varm[key_added + '_loadings'] = dimred_info['loadings']

##### DEPRECATED, left in temporarily to ensure new method outputs match old method outputs. ######

def reduce_old(X, n_comps=50, iters=1, max_bins=float('inf'), fast_version=True, scaled_bins=False, nbhds=None,
           rep=None, nbhd_size=15, n_pcs=50, metric='euclidean', model='wilcoxon',
           keep_scores=False, keep_loadings=False, keep_all_iters=False, verbose=False, n_tests = 'auto',
           seed=10,  **kwargs):
    """
    DEPRECATED: please use reduce.

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
    if n_tests == 'auto':
        n_tests = bootstrapped_ntests(X, model=model, k=nbhd_size, seed=seed, **kwargs)
        if verbose:
            print('multi-testing correction for {} features'.format(n_tests))

    result = {}  # final result, if return_all_iters=True

    if nbhds is None:
        if rep is None:
            a, bt, c, d = sc.tl.pca(X, n_comps=n_pcs, return_info=True)
            rep =  X @ bt.transpose()

            #rep = sc.tl.pca(X, n_comps=n_pcs)

        nn = NearestNeighbors(n_neighbors=nbhd_size, metric=metric)
        nn.fit(rep)

        nbhds = nn.kneighbors(return_distance=False)

        # add points to their own neighborhoods
        nbhds = np.concatenate((np.array(range(X.shape[0])).reshape(-1, 1), nbhds), axis=1)

    for i in range(iters):
        if verbose:
            print('\niteration {}'.format(i + 1))

        if model != 'binomial': #don't return bin info
            infos = info_score(X, nbhds,
                                                        max_bins=max_bins,
                                                        return_bin_info=True,
                                                        fast_version=fast_version,
                                                        verbose=verbose,
                                                        scaled=scaled_bins,
                                                        model=model,
                                                        n_tests=n_tests,
                                                        **kwargs)
        elif i == 0:
            # keep binomial scores for future runs
            infos, gene_bins, binom_scores = info_score(X, nbhds,
                                                        max_bins=max_bins,
                                                        return_bin_info=True,
                                                        fast_version=fast_version,
                                                        verbose=verbose,
                                                        scaled=scaled_bins,
                                                        model=model,
                                                        n_tests=n_tests,
                                                        **kwargs)
        else:
            infos = info_score(X, nbhds,
                               max_bins=max_bins,
                               binom_scores=binom_scores,
                               gene_bins=gene_bins,
                               fast_version=fast_version,
                               verbose=verbose,
                               scaled=scaled_bins,
                               model=model,
                               n_tests=n_tests,
                               **kwargs)


        a, bt, c, d = sc.tl.pca(infos, n_comps=n_comps, return_info=True)

        # pca(infos, k=n_comps, raw=True, pc_iters=pc_iters)

        current_dimred = X @ bt.transpose()  # metagene expression values in X

        # if scale_scs:
        #     norms = infos @ csr_matrix(bt.transpose())
        #     norms = sparse.linalg.norm(norms, axis=0, ord=2)
        #     print(norms)
        #     print(current_dimred.shape)
        #     print(norms.shape)
        #     current_dimred = current_dimred * norms
        if keep_all_iters:
            result['reduction_{}'.format(i + 1)] = current_dimred

        # compute neighborhoods for next iteration, using current reduction
        if i < (iters - 1):
            nn = NearestNeighbors(n_neighbors=nbhd_size, metric=metric)
            nn.fit(current_dimred)

            nbhds = nn.kneighbors(return_distance=False)
            nbhds = np.concatenate((np.array(range(X.shape[0])).reshape(-1, 1), nbhds), axis=1)

    if not keep_scores and not keep_loadings and not keep_all_iters:
        return current_dimred

    if not keep_all_iters:
        result['reduction'] = current_dimred  # store result
    if keep_scores:
        result['scores'] = csr_matrix(infos)
    if keep_loadings:
        result['loadings'] = bt.transpose()

    return result