"""a bunch of functions to generate fake data"""
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc

def pancakes(n_pts=100, corr=1, n_cor=2, n_noise=2, minor_fraction=0.5):
    split = [1]*int(np.floor(n_pts*minor_fraction)) + [0]*int(np.ceil(n_pts *(1-minor_fraction)))

    split_dims = np.tile(split, (n_cor,1))

    noisy_dims = np.random.choice([0,1], size=(n_noise, n_pts))
    mat = np.transpose(np.concatenate((split_dims, noisy_dims), axis=0))


    adata = sc.AnnData(csr_matrix(mat))
    adata.obs['label'] = [str(x) for x in split]

    return adata


