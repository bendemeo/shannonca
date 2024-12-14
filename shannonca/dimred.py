from .score import bootstrapped_ntests
from .scorers import Scorer, WilcoxonScorer, BinomialScorer, ChunkedScorer, TScorer
from .correctors import FWERCorrector
from .embedders import SCAEmbedder
from .connectors import MetricConnector, Connector

def reduce(X, n_comps=50, iters=1, nbhds=None,
           nbhd_size=15,  metric='euclidean', model='wilcoxon',
           keep_scores=False, keep_loadings=False, keep_all_iters=False, verbose=False, n_tests = 'auto',
           seed=10,  chunk_size=None, **kwargs):
    """
    Compute an SCA reduction of the input data.

    Parameters
    ----------
    X : numpy.ndarray or scipy.spmatrix
    (num cells)x(num_genes)-sized array or sparse matrix to be dimensionality-reduced.
    n_comps : int, optional
    Desired dimensionality of the reduction. Default is 50.
    iters : int, optional
    Number of iterations of SCA. More iterations usually strengthen the signal, stabilizing around 3-5. Default is 1.
    nbhd_size : int, optional
    Size of neighborhoods used to assess the local expression of a gene. Should be smaller than the smallest subpopulation; default is 15.
    model : str, optional
    Model used to test for local enrichment of genes, used to compute information scores. One of ["wilcoxon", "binomial", "ttest"], default "wilcoxon" (recommended).
    nbhds : numpy.ndarray or list, optional
    Optional - if k-neighborhoods of points are already determined, they can be specified here as a (num_cells)*k array or list. Otherwise, they will be computed from the PCA embedding. Default is None.
    metric : str, optional
    Metric used to compute k-nearest neighbor graphs for SCA score computation. Default is "euclidean". See sklearn.neighbors.DistanceMetric for list of choices.
    keep_scores : bool, optional
    If True, keep and return the information score matrix. Default is False.
    keep_loadings : bool, optional
    If True, returns loadings of each gene in each metagene as a dense matrix. Default is False.
    verbose : bool, optional
    If True, print progress. Default is False.
    n_tests : str or int, optional
    Effective number of independent genes per cell, used for FWER multiple testing correction. Set to "auto" to automatically determine by bootstrapping. Default is "auto".
    kwargs : dict, optional
    Other arguments to be passed to the chosen scorer.

    Returns
    -------
    numpy.ndarray or dict
    If keep_scores or keep_loadings are both False, a (n cells)x(n_comps)-dimensional array of reduced features.
    Otherwise, a dictionary with keys 'reduction', 'scores' and/or 'loadings'.
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
                  iters=1, model='wilcoxon', **kwargs):
    """
    Compute and store an SCA reduction of the input data in an AnnData object.

    Parameters
    ----------
    adata : scanpy.AnnData
        AnnData object containing single-cell transcriptomic data to be reduced.
    keep_scores : bool, optional
        If True, stores information score matrix in `adata.layers[key_added+'_score']`. Default is False.
    keep_loadings : bool, optional
        If True, stores loadings in `adata.varm[key_added+'_loadings']`. Default is False.
    keep_all_iters : bool, optional
        If True, store the embedding after each iteration in `adata.obsm[key_added+'_'+i]` for i in 1,2,...iters. Default is False.
    layer : str or None, optional
        Layer to reduce. If None, reduces `adata.X`. Otherwise, reduces `adata.layers[layer]`. Default is None.
    key_added : str, optional
        Namespace for storage of results. Defaults to 'sca'.
    iters : int, optional
        Number of SCA iterations to run. Default is 1.
    model : str, optional
        Model used to test for local enrichment of genes, used to compute information scores. One of ["wilcoxon","binomial","ttest"], default "wilcoxon" (recommended).
    **kwargs : dict
        Additional arguments to be passed to ``reduce`` (e.g. verbose, n_tests, chunk_size).

    Returns
    -------
    None
        The function modifies the `adata` object in place, adding the reduction results to `adata.obsm`, `adata.layers`, and `adata.varm` as specified by the parameters.
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