from scanpy.tools import pca
from .correctors import FWERCorrector
from .utils import bootstrapped_ntests

class Embedder:
    def __init__(self):
        pass

    def embed(self,X):
        #basic embedder does absolutely nothing
        return(X)

class SVDEmbedder(Embedder):
    #Project input to its right singular vectors

    def __init__(self, n_comps=50, svd_mat = None, **kwargs):
        # svd_mat is the matrix whose singular vectors we project the input to; if none, project data to its own SVDs.
        #additional arguments passed to sc.tl.pca

        self.n_comps = n_comps
        self.kwargs = kwargs
        self.svd_mat = svd_mat

    def embed(self, X, keep_loadings=False):
        if self.svd_mat is None:
            self.svd_mat = X
        a, bt, c, d = pca(self.svd_mat, n_comps=self.n_comps, return_info=True, **self.kwargs)

        if keep_loadings:
            self.loadings = bt.transpose()
        return X @ bt.transpose()


class SCAEmbedder(Embedder):
    def __init__(self, scorer, connector=None, nbhds=None, n_comps=50, iters=1):
        self.nbhds = nbhds
        self.n_comps = n_comps
        self.iters = iters
        self.scorer = scorer
        self.connector = connector

        assert((connector is not None) or (nbhds is not None), "must specify connector or neighborhoods")

    def embed(self, X, keep_scores=False, keep_loadings=False, keep_all_iters=False):
        if self.iters == 0:
            # base case: just do PCA

            embedder = SVDEmbedder(n_comps=self.n_comps)

            if keep_all_iters:
                self.embedding_dict = {}

            return embedder.embed(X)

        else:
            base_embedder = SCAEmbedder(scorer=self.scorer, connector=self.connector,
                                        nbhds=self.nbhds, n_comps=self.n_comps, iters=self.iters-1)

            base_embedding = base_embedder.embed(X, keep_scores=False, keep_loadings=False, keep_all_iters=keep_all_iters)

            #make nbhds if not provided
            if self.nbhds is None or self.iters > 1:
                self.nbhds = self.connector.connect(base_embedding)

            scores = self.scorer.score(X, self.nbhds) #compute info scores

            final_embedder = SVDEmbedder(n_comps=self.n_comps, svd_mat=scores)

            result = final_embedder.embed(X, keep_loadings=True)

            if keep_scores:
                self.scores = scores
            if keep_loadings:
                self.loadings = final_embedder.loadings
            if keep_all_iters:
                self.embedding_dict = base_embedder.embedding_dict
                self.embedding_dict[self.iters] = result

            return(result)






