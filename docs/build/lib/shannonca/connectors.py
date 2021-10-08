from sklearn.neighbors import NearestNeighbors
import numpy as np

class Connector:
    def __init__(self):
        pass

    def connect(self,X):
        # insert code to make nearest neighbor graphs
        # should return a LIST of neighborhoods
        pass

class MetricConnector(Connector):
    def __init__(self, n_neighbors=15, metric='euclidean', include_self=True):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.include_self=include_self

    def connect(self, X):
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        nn.fit(X)
        nbrs = nn.kneighbors(return_distance=False)


        if self.include_self:
            nbrs = np.concatenate((np.array(range(X.shape[0])).reshape(-1, 1), nbrs), axis=1)

        return(nbrs.tolist())




