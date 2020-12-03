### stuff that adds dimensionality reductions when called on a scanpy object
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from .wts import info_score
import numpy as np
from fbpca import pca
import pandas as pd

def info_pca(nbhd_size=15, n_neighbors=15, n_init_pcs=50, n_info_pcs=50, key_added='info_pca', metric='cosine',
             max_bins=float('inf'), iters=1, two_tailed=True, entropy_normalize=False, p_val=True,
             keep_scores=True, binarize=True, **kwargs):



    def f(d):
        sc.tl.pca(d, n_comps=n_init_pcs)

        if binarize:
            U,s,Vt = pca((d.X>0).astype('float'), k=n_init_pcs)
            pcs = U*s
        else:
            sc.tl.pca(d.X, n_comps=n_init_pcs)
            pcs = d.obsm['X_pca']


        nn = NearestNeighbors(n_neighbors=nbhd_size);
        nn.fit(pcs)
        dist, nbhds = nn.kneighbors()

        # add points to their own neighborhoods
        nbhds = np.concatenate((np.array(range(d.shape[0])).reshape(-1, 1), nbhds), axis=1)

        for i in range(iters):
            print('iteration {}'.format(i))

            infos = info_score((d.X>0).astype('float'), nbhds, weighted_dir=True, max_bins=max_bins, two_tailed=two_tailed,
                               entropy_normalize=entropy_normalize, p_val=p_val, **kwargs)

            U, s, Vt = pca(infos, k=n_info_pcs, raw=True)

            if binarize:
                X = (d.X>0).astype('float').todense()
                d.obsm[key_added] = np.matmul(X, Vt.transpose())
            else:
                d.obsm[key_added] = np.matmul(d.X.todense(), Vt.transpose())


            d.varm[key_added+'_loadings'] = Vt.transpose()

            print(d.obsm[key_added].shape)

            nn = NearestNeighbors(n_neighbors=nbhd_size);
            nn.fit(d.obsm[key_added])

            dist, nbhds = nn.kneighbors()
            nbhds = np.concatenate((np.array(range(d.shape[0])).reshape(-1, 1), nbhds), axis=1)

        sc.pp.neighbors(d, use_rep=key_added, key_added=key_added, metric=metric, n_neighbors=n_neighbors)

        if keep_scores:
            info_df = pd.DataFrame(infos)
            info_df.columns = d.var.index
            info_df.index = d.obs.index
            d.layers[key_added+'_score'] = info_df
    return f