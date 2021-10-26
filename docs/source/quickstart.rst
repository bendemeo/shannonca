Quick Start 
=======================

shannonca's [top-level API](https://shannonca.readthedocs.io/en/latest/api.html) offers two functions for use with and without scanpy. [`shannonca.dimred.reduce`](https://shannonca.readthedocs.io/en/latest/generated/shannonca.dimred.reduce.html#shannonca.dimred.reduce) accepts an array or sparse matrix, and outputs a lower-dimensional SCA reduction. Optional additional parameters control the dimensionality of the output, the neighborhood size, the model used to assess over- or under-expression, the degree of multiple testing correction when computing information scores, and various other settings. 

A simple use case:

.. code-block:: python

  from scipy.io import mmread
  from shannonca.dimred import reduce

  X = mmread('my_sparse_matrix.mtx')  # read in a high-dimensional, sparse matrix

  # run SCA for 5 iterations to produce a 50-dimensional embedding
  X_dimred = reduce(X, n_comps=50, iters=5)


If we specify ``keep_scores=True``, the return type is a dictionary, with the information score matrix keyed by  "scores" and the reduced dataset keyed by "reduction". If ``keep_loadings=True``, this dictionary also stores the loadings of each Shannon component at "loadings". If ``keep_all_iters=True``, it additionally stores intermediate dimensionality reductions after each iteration, keyed by "reduction_(iteration number)". This is useful for testing the effect of more iterations.

Scanpy integration
=========================

For scanpy users, we offer a wrapper ``reduce_scanpy``. This function accepts a scanpy ``AnnData`` object ``adata``, and runs ``reduce`` on the specified ``layer`` (by default, on ``adata.X``). The resulting reduction is stored under ``adata.obsm[key_added]``, where ``key_added`` defaults to "sca". Otherwise, the parameters are the same as ``reduce``. If ``keep_scores`` is True, the scores are stored at ``adata.layers[key_added+'_score']``. If ``keep_loadings`` is True, loadings are added in ``.varm[key_added+'loadings']``. 
