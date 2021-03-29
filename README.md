# SCA (Shannon Components Analysis)

Shannon Components Analysis (formerly called scalpel) is a dimensionality reduction technique for single-cell data which leverages mathematical information theory to identify biologically informative axes of variation in single-cell transcriptomic data, enabling recovery of rare and common cell types at superior resolution. It is written in Python. The pre-print can be found [here](https://www.biorxiv.org/content/10.1101/2021.01.19.427303v1).

Required packages: scanpy, sklearn, numpy, scipy, fbpca, itertools

## Usage
### Dimensionality Reduction
SCA generates information score matrices, which are used to generate linear combinations of genes (metagenes) that are biologically informative. The package includes workflows both with and without Scanpy under `sca.dimred`.

#### Without Scanpy
The `reduce` function accepts a (num genes) x (num cells) matrix X, and outputs a dimensionality-reduced version with fewer features. The input matrix may be normalized or otherwise processed, but a zero in the input matrix must indicate zero recorded transcripts.
```
from sca.dimred import reduce

X = mmread('mydata.mtx').transpose() # read some dataset

reduction = reduce(X, n_comps=50, n_pcs=50, iters=1, max_bins=float('inf'))
```
`reduction` is an (num cells) x (comps)-dimensional matrix. The function optionally returns SCA's score matrix (if keep_scores=True), metagene loadings (if keep_loadings=True), or intermediate results (if iters>1 and keep_all_iters=True). If at least one of these is returned, the return type is a dictionary with keys for 'reduction', 'scores', and 'loadings'. If keep_all_iters=True, the reductions after each iteration will be keyed by 'reduction_i' for each iteration number i. 

The starting representation by default is a PCA representation with 50 components, but the user can specify their own using `rep`. Starting neighborhoods are computed by default using cosine distance in this representation (controlled by `metric`). See the docstring for more detailed and comprehensive parameter descriptions.

#### With Scanpy
Scanpy (https://github.com/theislab/scanpy) is a commonly-used single-cell workflow. To compute a reduction in place on a scanpy AnnData object, use `reduce_scanpy`:
```
import scanpy as sc
from sca.dimred import reduce_scanpy

adata = sc.AnnData(X)
reduce_scanpy(adata, keep_scores=True, keep_loadings=True, keep_all_iters=True, layer=None, key_added='sca', iters=1, n_comps=50, n_pcs=50)
```
This function shares all parameters with `reduce`, but instead of returning the output, it updates the input AnnData object. Dimensionality reductions are stored in `adata.obsm[key_added]`, or, if keep_all_iters=True, in `adata.obsm['key_added_i']` for each iteration number i. If `keep_scores=True` in the reducer constructor, the information scores of each gene in each cell are stored in `adata.layers[key_added]`. If `layer=None`, the algorithm is run on `adata.X`; otherwise, it is run on `adata.layers[layer]`.

#### [Deprecated] Curried version

First, construct a reducer using the `info_pca` function.

```
from sca.dimred import info_pca
reducer = info_pca(nbhd_size=15, n_neighbors=15, n_init_pcs=50, n_info_pcs=50, key_added='info_pca', metric='cosine',
             max_bins=float('inf'), iters=1)
```
Shown are the default parameters. `nbhd_size` determines the size of local neighborhoods used in comparing local vs. global expression of each gene. `n_neighbors` determines the number of neighbors in the NN graph computed at the end. `n_init_pcs` determines how many PCs to use in computing initial locales. `n_info_pcs` determines the dimensionality of the final reduction. `iters` determines the number of iterations; more iterations generally gives results more different from PCA.

The resulting reducer can be called on a scanpy object `adata`, which adds information in place. Simply run
```
reducer(adata)
```
Dimensionality reductions are stored in `adata.obsm[key_added]`. If `keep_scores=True` in the reducer constructor, the information scores of each gene in each cell are stored in `adata.layers[key_added]`. Finally, the `n_neighbors`-nearest neighbor graph in the sca representation under the specified metric is stored in `adata.uns[key_added]'.

### Batch correction
SCA implements batch correction via selection of genes that have high within-batch information scores. The procedure is implemented  in the `sca.integrate` submodule by `correct` and `correct_scanpy`. The former takes a list of matrices from each batch, whereas the latter takes an AnnData object with a column indicating batch in `.obs`. They return tuples containing the corrected data, and the list of features retained.
