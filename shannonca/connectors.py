from sklearn.neighbors import NearestNeighbors
import numpy as np
from .embedders import Embedder
from .lknn import multires_dists, knn_cover
from .utils import dist_mat_to_nbhds


class Connector:
    def __init__(self):
        pass

    def connect(self,X, **kwargs):
        # insert code to make nearest neighbor graphs
        # should return a LIST of neighborhoods
        pass


class MetricConnector(Connector):
    def __init__(self, n_neighbors=15, metric='euclidean', include_self=True):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.include_self=include_self

    def connect(self, X, **kwargs):
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, **kwargs)
        nn.fit(X)
        nbrs = nn.kneighbors(return_distance=False)


        if self.include_self:
            nbrs = np.concatenate((np.array(range(X.shape[0])).reshape(-1, 1), nbrs), axis=1)

        return(nbrs.tolist())

class MultiResConnector(Connector):
    def __init__(self, base_embedder, base_metric='euclidean', sub_embedder = None, sub_metric=None, verbose=False):
        self.base_embedder = base_embedder
        self.base_metric = base_metric
        self.verbose = verbose
        if sub_embedder is None:
            sub_embedder = base_embedder
        self.sub_embedder = sub_embedder
        if sub_metric is None:
            sub_metric = base_metric
        self.sub_metric = sub_metric
        self.sub_connector = sub_metric

    def connect(self, X, resolution, n_neighbors=15, aggregator='max', max_neighbors=float('inf'), n_covers=1,
                cover=None, return_dists=False, return_mat=False, **kwargs):
        # compute cover based on global embedding
        if cover is None:
            global_embedding = self.base_embedder.embed(X)
            cover = knn_cover(global_embedding, metric=self.base_metric, k=resolution, n_covers=n_covers)
        dist_mat = multires_dists(X, covering_sets=cover, embedder=self.sub_embedder,
                                  metric=self.sub_metric, aggregator=aggregator, max_neighbors=max_neighbors)

        if return_mat:
            return(dist_mat)
        nbhds, dists = dist_mat_to_nbhds(dist_mat, k=n_neighbors)
        if return_dists:
            return((nbhds, dists))
        else:
            return(nbhds)




