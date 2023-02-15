from scanpy.tools import pca
from .correctors import FWERCorrector

class Embedder:
    """
    Base class for Embedders. Does nothing, acting as an "identity" embedder.
    """
    def __init__(self):
        """
        Constructor, extended by child classes.
        """
        pass

    def embed(self,X):
        """
        Returns the input data, unchanged.
        """
        return(X)

class SVDEmbedder(Embedder):
    """
    Embedder that projects data to its top right singular vectors.
    """

    def __init__(self, n_comps=50, svd_mat = None, **kwargs):
        """
        Constructor

        :param n_comps: Number of top singular vectors to keep. Defines output dimensionality.
        :type n_comps: int
        :param svd_mat: if provided, project the input to the right singular vectors of this matrix. If None (default), project to singular vectors of the input.
        :type svd_mat: numpy.ndarray | matrix | spmatrix
        :param kwargs: Additional arguments passed to scanpy.tl.pca
        """

        self.n_comps = n_comps
        self.kwargs = kwargs
        self.svd_mat = svd_mat

    def embed(self, X, keep_loadings=False):
        """
        Project X onto a set of top right singular vectors, and return the result

        :param X: Data to be reduced
        :type X: np.ndarray | matrix | spmatrix
        :param keep_loadings: Whether to keep the loading vectors. If True, the loading vectors are stored as self.loadings. Default False.
        :type keep_loadings: bool

        :return: A low-dimensional representation of X.
        :rtype: np.ndarray | matrix | spmatrix
        """
        if self.svd_mat is None:
            self.svd_mat = X
        a, bt, c, d = pca(self.svd_mat, n_comps=self.n_comps, return_info=True, **self.kwargs)

        self.variance_explained = d
        
        if keep_loadings:
            self.loadings = bt.transpose()
        return X @ bt.transpose()



class SCAEmbedder(Embedder):
    """
    Embedder that performs SCA on the input.

    """
    def __init__(self, scorer, connector=None, n_comps=50, iters=1):
        """
        Constructor.

        :param n_comps: Number of Shannon components to compute
        :type n_comps: int
        :param scorer: Scorer object used to generate scores from neighborhoods
        :type scorer: Scorer
        :param connector: Connector object used to generate neighborhoods of the input data. If None, you must provide neighborhoods when running ``embed``.
        :type connector: Connector | None
        :param iters: Number of SCA iterations to run
        :type iters: int

        """
        self.n_comps = n_comps
        self.iters = iters
        self.scorer = scorer
        self.connector = connector
        self.embedding_dict = {}

    def embed(self, X, keep_scores=False, keep_loadings=False, keep_all_iters=False, nbhds=None, verbose=True):
        """
        Compute the SCA embedding

        :param X: Input data, with one row per observation and one column per feature.
        :type X: np.ndarray | matrix | spmatrix
        :param keep_scores: If True, SCA scores are retained in the .scores attribute of this object. Default False.
        :type keep_scores: bool
        :param keep_loadings: If True, Shannon Component loadings are retained in the .loadings attribute. Default False.
        :type keep_loadings: bool
        :param keep_all_iters: If True, intermediate embeddings after each iteration are retained in self.embedding_dict, a dictionary of embeddings keyed by iteration number. Default False.
        :type keep_all_iters: bool
        :param nbhds: (Optional) Neighborhoods to use when computing scores. Overrides the Connector, if specified at construction. If Connector=None at construction, this parameter is not optional.
        :type nbhds: list | None
        :param verbose: If True, print progess during execution. Default True
        :type verbose: bool

        """
        assert (self.connector is not None) or (nbhds is not None), "must specify connector or neighborhoods"

        if self.iters == 0:
            # base case: just do PCA

            embedder = SVDEmbedder(n_comps=self.n_comps)

            if keep_all_iters:
                self.embedding_dict = {}
               
            result = embedder.embed(X)
            self.variance_explained = embedder.variance_explained
            
            return result


        else:
            #make nbhds if not provided
            if nbhds is None or self.iters > 1:
                base_embedder = SCAEmbedder(scorer=self.scorer, connector=self.connector,
                                            n_comps=self.n_comps, iters=self.iters - 1)

                base_embedding = base_embedder.embed(X, keep_scores=False, keep_loadings=False,
                                                     keep_all_iters=keep_all_iters, nbhds=nbhds)
                self.embedding_dict = base_embedder.embedding_dict
                if verbose:
                    print('iteration {}'.format(self.iters))
                nbhds = self.connector.connect(base_embedding)
            else:
                self.embedding_dict = {}

            scores = self.scorer.score(X, nbhds) #compute info scores

            final_embedder = SVDEmbedder(n_comps=self.n_comps, svd_mat=scores)

            result = final_embedder.embed(X, keep_loadings=True)
            
            self.variance_explained = final_embedder.variance_explained

            if keep_scores:
                self.scores = scores
            if keep_loadings:
                self.loadings = final_embedder.loadings
            if keep_all_iters:
                self.embedding_dict[self.iters] = result

            return(result)






