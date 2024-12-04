from sklearn.neighbors import NearestNeighbors
import numpy as np
from .embedders import Embedder
from .lknn import multires_dists, knn_cover
from .utils import dist_mat_to_nbhds


class Connector:
    """
    Base class for Connectors.
    """
    def __init__(self):
        pass

    def connect(self,X, **kwargs):
        """
        Method to generate nearest neighbors of input data.
        """
        # insert code to make nearest neighbor graphs
        # should return a LIST of neighborhoods
        pass


class MetricConnector(Connector):
    """
    Make neighborhoods of observations using a metric on the input data.
    """
    def __init__(self, n_neighbors=15, metric='euclidean', include_self=True):
        """
        Constructor

        :param n_neighbors: number of neighbors of each observation to compute
        :type n_neighbors: int
        :param metric: Metric to use. See `DistanceMetric <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric>`_ for available metrics. Default 'euclidean'.
        :type metric: str
        :param include_self: whether observations should be classified as their own neighbors. Default True.
        :type include_self: bool

        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.include_self=include_self

    def connect(self, X, **kwargs):
        """
        Make neighborhoods of the input

        :param X: input data, with one row per observation and one column per feature.
        :type X: np.ndarray | matrix | spmatrix
        :param kwargs: Additional arguments passed to sklearn.neighbors.NearestNeighbors

        :return: A list of neighbor indices for each observation. Indexes rows of the input matrix
        :rtype: list

        """
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, **kwargs)
        nn.fit(X)
        nbrs = nn.kneighbors(return_distance=False)


        if self.include_self:
            nbrs = np.concatenate((np.array(range(X.shape[0])).reshape(-1, 1), nbrs), axis=1)

        return(nbrs.tolist())
