"""a bunch of functions to generate fake data"""
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc

def pancakes(n_pts=100, n_cor=2, n_anticor=2, n_noise=2, minor_fraction=0.5, random_fraction=0.5, marker_fraction=0.9, noise_level=0.1, dropout_clust=True):

    split =  [0]*int(np.ceil(n_pts *(1-minor_fraction))) + [1]*int(np.floor(n_pts*minor_fraction))

    markers_on = np.random.choice([0,1], size=(np.floor(n_pts*minor_fraction).astype('int'), n_cor), p = [ 1-marker_fraction, marker_fraction])
    markers_off = np.random.choice([0,1], size=(np.ceil(n_pts *(1-minor_fraction)).astype('int'), n_cor), p = [1-noise_level, noise_level])

    split_dims = np.vstack((markers_on, markers_off))
    #split_dims = np.tile(split, (n_anticor,1))


    normal_off = np.random.choice([0,1], size=(np.floor(n_pts*minor_fraction).astype('int'), n_cor), p = [ marker_fraction, 1-marker_fraction])
    normal_on = np.random.choice([0,1], size=(np.ceil(n_pts *(1-minor_fraction)).astype('int'), n_cor), p = [noise_level, 1-noise_level])

    split_dims = np.vstack((markers_on, markers_off))
    off_dims = np.vstack((normal_on, normal_off))

    # off =  [0]*int(np.floor(n_pts*minor_fraction)) + [1]*int(np.ceil(n_pts *(1-minor_fraction)))
    # off_dims = np.tile(off, (n_cor, 1)).transpose()



    noisy_dims = np.random.choice([0,1], size=(n_noise, n_pts), p=[1-random_fraction, random_fraction]).transpose()
    if dropout_clust:
        mat = np.concatenate((split_dims, off_dims, noisy_dims), axis=1)
    else:
        mat = np.concatenate((split_dims, noisy_dims), axis=1)
    #print(mat.shape)

    adata = sc.AnnData(csr_matrix(mat))

    labs = np.array(['normal']*adata.shape[0])
    labs[:np.floor(n_pts*minor_fraction).astype('int')] = 'rare'
    if dropout_clust:
        labs[-1*np.floor(n_pts*minor_fraction).astype('int'):] = 'dropout'
    #print(len(labs))

    adata.obs['label'] = labs
    #adata.obs['label'] = [str(x) for x in np.array(split)+np.array(off)]

    return adata


