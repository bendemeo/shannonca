import numpy as np
from fbpca import pca
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scalpel.deprecated.embed import gene_info
from scipy.spatial.distance import euclidean

#functions that take a count matrix and produce a nearest-neighbor graph

def pca_nn(n_pcs=50, n_nbrs=15):

    def f(X):
        U,s,Vt = pca(X, k=n_pcs); X_dimred = U*s

        nn = NearestNeighbors(n_neighbors=n_nbrs)
        nn.fit(X_dimred)

        return nn.kneighbors_graph(X_dimred)

    return f

def scaled_nn(n_pcs = 50, n_nbrs=15, k=5, return_scale = False):

    def f(X):
        U,s,Vt = pca(X, k=n_pcs); X_dimred = U*s

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_dimred)

        dists, nbrs = nn.kneighbors(X_dimred)

        print(dists)
        dist_scales = dists[:,-1]

        print(np.min(dist_scales))

        pairwise_dists = pairwise_distances(X_dimred) #likely slow/memory intensive


        # scaled_pairwise_dists = np.divide(pairwise_dists,
        #                                   np.tile(dist_scales.reshape(-1,1), (1, pairwise_dists.shape[1])))

        scaling_factors = np.sqrt(np.outer(dist_scales, dist_scales))
        scaled_pairwise_dists = np.divide(pairwise_dists, scaling_factors)

        nn_scaled = NearestNeighbors(n_neighbors=n_nbrs, metric='precomputed')

        nn_scaled.fit(scaled_pairwise_dists)

        G = nn_scaled.kneighbors_graph(scaled_pairwise_dists)

        if return_scale:
            return(G, dist_scales)
        else:
            return G
    return f

def direct_pairwise(n_neighbors=15, metric='euclidean'):

    def f(X):
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(X)
        return nn.kneighbors_graph(X)

    return f


def info_weighted_pairwise(n_neighbors=15,
                           metric=euclidean,
                           info_nbrs = 20):
    def f(X):

        info_scores = gene_info(n_neighbors=info_nbrs, n_pcs=50)(X)

        scaled_distances = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            print('\r{}/{}'.format(i, X.shape[0]), end='   ')
            for j in range(X.shape[0]):
                feature_wts = info_scores[i, :] + info_scores[j, :]
                expr_i = np.array(X[i, :].todense()).flatten()
                expr_j = np.array(X[j, :].todense()).flatten()

                dist = metric(np.multiply(expr_i, feature_wts), np.multiply(expr_j, feature_wts))
                scaled_distances[i, j] = dist

        nn = NearestNeighbors(metric='precomputed')
        nn.fit(scaled_distances)
        return(nn.kneighbors_graph(n_neighbors=n_neighbors))

    return f






