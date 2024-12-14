# SCA (Surprisal Component Analysis)

Surprisal Component Analysis (SCA) is a dimensionality reduction technique for single-cell data which leverages mathematical information theory to identify biologically informative axes of variation in single-cell transcriptomic data, enabling recovery of rare and common cell types at superior resolution. It is written in Python. The pre-print can be found [here](https://www.biorxiv.org/content/10.1101/2021.01.19.427303v1).

For full documentation of shannonca's API, see our [Readthedocs page](https://shannonca.readthedocs.io/). 

## Installation

SCA is available via pip:
```pip install shannonca```

### Dependencies
SCA requires the following packages:

* scikit-learn
* scipy
* numpy
* matplotlib
* pandas
* seaborn
* scanpy


## Usage
### Dimensionality Reduction
SCA generates information score matrices, which are used to generate linear combinations of genes (metagenes) that are biologically informative. The package includes workflows both with and without Scanpy under `sca.dimred`.

#### Without Scanpy
The `reduce` function accepts a (num genes) x (num cells) matrix X, and outputs a dimensionality-reduced version with fewer features. The input matrix may be normalized or otherwise processed, but a zero in the input matrix must indicate zero recorded transcripts.
```
from shannonca.dimred import reduce

X = mmread('mydata.mtx').transpose() # read some dataset

reduction = reduce(X, n_comps=50, n_pcs=50, iters=1, nbhd_size=15, metric='euclidean', model='wilcoxon', chunk_size=1000, n_tests='auto')
```
`reduction` is an (num cells) x (`n_comps`)-dimensional matrix. The function optionally returns SCA's score matrix (if keep_scores=True), metagene loadings (if keep_loadings=True), or intermediate results (if iters>1 and keep_all_iters=True). If at least one of these is returned, the return type is a dictionary with keys for 'reduction', 'scores', and 'loadings'. If keep_all_iters=True, the reductions after each iteration will be keyed by 'reduction_i' for each iteration number i. 

Starting neighborhoods are computed by default using Euclidean distance (controlled by `metric`) in `n_comps`-dimensional PCA space. See the docstring for more detailed and comprehensive parameter descriptions.

#### With Scanpy
Scanpy (https://github.com/theislab/scanpy) is a commonly-used single-cell workflow. To compute a reduction in place on a scanpy AnnData object, use `reduce_scanpy`:
```
import scanpy as sc
from shannonca.dimred import reduce_scanpy

adata = sc.AnnData(X)
reduce_scanpy(adata, keep_scores=True, keep_loadings=True, keep_all_iters=True, layer=None, key_added='sca', iters=1, n_comps=50)
```
This function shares all parameters with `reduce`, but instead of returning the reduction, it updates the input AnnData object. Dimensionality reductions are stored in `adata.obsm[key_added]`, or, if keep_all_iters=True, in `adata.obsm['key_added_i']` for each iteration number i. If `keep_scores=True` in the reducer constructor, the information scores of each gene in each cell are stored in `adata.layers[key_added_score]`. If `layer=None`, the algorithm is run on `adata.X`; otherwise, it is run on `adata.layers[layer]`.

## Troubleshooting
If you are having trouble running SCA, try the following:

* Pull from the github repository to ensure that your version of SCA is up to date. 
* Ensure that the Python version is at least 3.0, and that the installations of scanpy, numpy, scipy, and sklearn are up to date. 
* When running the `reduce` function, ensure that the input is either a CSR sparse matrix (`scipy.sparse.csr_matrix`) or a dense numpy array, with one row per cell and one column per gene. Coercion to sparse matrices is easy via `scipy.sparse.tocsr()`
* When running `reduce_scanpy`, ensure that the input is a scanpy anndata object. 
* Ensure that the data type of the input is either an integer or float. 
* Double-check that the code follows the docstring for the relevant function: reduce or reduce_scanpy.
