import unittest
from shannonca.embedders import *
from shannonca.generate import pancakes
from shannonca.scorers import *
from shannonca.correctors import *
from shannonca.connectors import *
from shannonca.utils import add_nbrs_from_dists, add_nbrs
from shannonca.dimred import reduce, reduce_old
import scanpy as sc
import numpy as np


class test_connectors(unittest.TestCase):

    def __init__(self,*args, **kwargs):
        super(test_connectors, self).__init__(*args, **kwargs)
        self.seed = 20
        np.random.seed(self.seed)
        self.testData = pancakes(n_pts=1000, n_cor=10, n_anticor=10, n_noise=1000)

    def test_MultiResConnector(self):

        base_embedder = SCAEmbedder(scorer=WilcoxonScorer(corrector=FWERCorrector(n_tests=100)),
                                    connector=MetricConnector(metric='euclidean', n_neighbors=15),
                                    n_comps=5)

        connector = MultiResConnector(base_embedder=base_embedder, base_metric='euclidean')

        nbhds = connector.connect(self.testData.X, resolution=50, n_neighbors=10)

        add_nbrs(self.testData, nbhds, dists=nbhds, key_added='multires')
        sc.tl.umap(self.testData, neighbors_key='multires')
        sc.pl.umap(self.testData, edges=True, neighbors_key='multires')

        return(nbhds)
