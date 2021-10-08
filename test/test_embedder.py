import unittest
from shannonca.embedders import *
from shannonca.generate import pancakes
from shannonca.scorers import *
from shannonca.correctors import *
from shannonca.connectors import *
from shannonca.dimred import reduce, reduce_old
import scanpy as sc
import numpy as np

class test_embedders(unittest.TestCase):

    def __init__(self,*args, **kwargs):
        super(test_embedders, self).__init__(*args, **kwargs)
        self.seed = 20
        np.random.seed(self.seed)
        self.testData = pancakes(n_pts=100, n_cor=10, n_anticor=10, n_noise=100).X

    def test_SVDEmbedder(self):
        embedder = SVDEmbedder(n_comps=50)
        embedding = embedder.embed(self.testData)
        self.assertTrue(embedding.shape == (self.testData.shape[0], 50))

    def test_SCAEmbedder(self):
        corrector = FWERCorrector(n_tests=10)
        scorer = WilcoxonScorer(corrector=corrector)
        connector = MetricConnector(n_neighbors=15, metric='euclidean', include_self=True)

        embedder = SCAEmbedder(scorer=scorer, connector=connector, n_comps=50, iters=5)

        result = embedder.embed(self.testData, keep_scores=True, keep_all_iters=True, keep_loadings=True)

        result_old = reduce_old(self.testData, n_comps=50, iters=5, n_tests=10, nbhd_size=15,
                            n_pcs=50, metric='euclidean', model='wilcoxon', keep_scores=True)

        # make sure it's the same as the old one.
        diff = np.max(np.abs(result-result_old['reduction']))
        self.assertTrue(diff <= 1e-10)

        print('hi')

        #they are not the same...

        # #make neighborhoods
        # adata_new = sc.AnnData(result)
        # sc.pp.neighbors(adata_new, metric='euclidean')
        # sc.tl.umap(adata_new)
        # sc.pl.umap(adata_new)
        #
        # adata_old = sc.AnnData(result_old)
        # sc.pp.neighbors(adata_old, metric='euclidean')
        # sc.tl.umap(adata_old)
        # sc.pl.umap(adata_old)
        #

    def test_reducers(self):
        #make sure new modular reducer matches old reducer in output.

        res_new = reduce(self.testData, n_comps=50, iters=5, n_tests=10, nbhd_size=10,
                         metric='euclidean', model='binomial', keep_scores=True,
                         keep_all_iters=True)

        res_old = reduce_old(self.testData, n_comps=50, iters=5, n_tests=10, nbhd_size=10,
                            n_pcs=50, metric='euclidean', model='binomial', keep_scores=True,
                             keep_all_iters=True)

