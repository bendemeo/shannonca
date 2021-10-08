from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances

def knn_cover(X, k=50, metric='euclidean',n_covers=1, seed=10, **kwargs):
    sets = []
    np.random.seed(seed)
    for i in range(n_covers):
        is_uncovered = np.array([True]*X.shape[0])

        nn = NearestNeighbors(metric=metric, n_neighbors=k, **kwargs)
        nn.fit(X)
        while np.sum(is_uncovered) > 0:
            seed = np.random.choice(np.arange(X.shape[0])[is_uncovered])
            new_set = nn.kneighbors(X[seed,:].reshape(1,-1), return_distance=False)
            new_set =np.append(new_set, seed) # add self to neighborhood
            sets.append(new_set.flatten())
            is_uncovered[new_set.flatten()] = False
    return sets

def multires_dists(X, covering_sets, embedder, metric='euclidean', aggregator='max',  max_neighbors=float('inf'), **kwargs):
    # d(x,y) is maximum distance in any covering set PCA

    distance_matrices = []
    final_matrix = csr_matrix((X.shape[0], X.shape[0]))

    # compute sparse pairwise distance matrices of each covering set
    for i, s in enumerate(covering_sets):
        print('set {}/{}'.format(i, len(covering_sets)), end='\r')
        X_sub = X[s.flatten(), :]

        X_sub_dimred = embedder.embed(X_sub, verbose=False)

        if max_neighbors == float('inf'):

            dists_sub = pairwise_distances(X_sub_dimred, metric=metric)

            cols = np.array(s.tolist() * len(s))
            rows = np.array([[i] * len(s) for i in s]).flatten()

            new_mat = csr_matrix((dists_sub.flatten(), (rows, cols)), shape=(X.shape[0], X.shape[0]))

        else:
            nn = NearestNeighbors(n_neighbors=max_neighbors, metric=metric)
            nn.fit(X_sub_dimred)

            dists_sub = nn.kneighbors_graph()

            rows, cols = dists_sub.nonzero()

            # in overall dataset
            global_rows = np.array(s)[rows]
            global_cols = np.array(s)[cols]

            new_mat = csr_matrix((dists_sub.data, (global_rows, global_cols)), shape=(X.shape[0], X.shape[0]))

        distance_matrices.append(new_mat)

        if aggregator == 'max':
            final_matrix = final_matrix.maximum(new_mat)

    return final_matrix

