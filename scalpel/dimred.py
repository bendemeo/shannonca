### stuff that adds dimensionality reductions when called on a scanpy object
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from .wts import info_score
import numpy as np
from fbpca import pca


def info_pca(nbhd_size=15, n_neighbors=15, n_init_pcs=50, n_info_pcs=50, key_added='info_pca', metric='cosine',
             max_bins=float('inf'), iters=1):


    def f(d):
        sc.tl.pca(d, n_comps=n_init_pcs)

        nn = NearestNeighbors(n_neighbors=nbhd_size);
        nn.fit(d.obsm['X_pca'])
        dist, nbhds = nn.kneighbors()


        # add points to their own neighborhoods
        nbhds = np.concatenate((np.array(range(d.shape[0])).reshape(-1, 1), nbhds), axis=1)

        for i in range(iters):
            print('iteration {}'.format(i))

            infos = np.power(info_score(d.X.todense(), nbhds, weighted_dir=True, max_bins=max_bins), 1)
            U, s, Vt = pca(infos, k=n_info_pcs, raw=True)

            d.obsm[key_added] = np.matmul(d.X.todense(), Vt.transpose())

            print(d.obsm[key_added].shape)

            nn = NearestNeighbors(n_neighbors=15);
            nn.fit(d.obsm[key_added])
            dist, nbhds = nn.kneighbors()
            nbhds = np.concatenate((np.array(range(d.shape[0])).reshape(-1, 1), nbhds), axis=1)

        sc.pp.neighbors(d, use_rep=key_added, key_added=key_added, metric=metric)

    return f