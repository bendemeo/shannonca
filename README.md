# scalpel

Scalpel is a dimensionality reduction technique for single-cell data that uses information-theoretic principles to accurately represent rare cell types. It is written in Python. 

Required packages: scanpy, sklearn, numpy, scipy, fbpca, itertools

## Usage

First, construct a reducer using the `info_pca` function. 

```
from scalpel.dimred import info_pca
reducer = info_pca(nbhd_size=15, n_neighbors=15, n_init_pcs=50, n_info_pcs=50, key_added='info_pca', metric='cosine',
             max_bins=float('inf'), iters=1)
```
Shown are the default parameters. `nbhd_size` determines the size of local neighborhoods used in comparing local vs. global expression of each gene. `n_neighbors` determines the number of neighbors in the NN graph computed at the end. `n_init_pcs` determines how many PCs to use in computing initial locales. `n_info_pcs` determines the dimensionality of the final reduction. `iters` determines the number of iterations; more iterations generally gives results more different from PCA. 

The resulting reducer can be called on a scanpy object `adata`, which adds information in place. Simply run 
```
reducer(adata)
```
Dimensionality reductions are stored in `adata.obsm[key_added]`. If `keep_scores=True` in the reducer constructor, the information scores of each gene in each cell are stored in `adata.layers[key_added]`. Finally, the `n_neighbors`-nearest neighbor graph in the scalpel representation under the specified metric is stored in `adata.uns[key_added]'. 

