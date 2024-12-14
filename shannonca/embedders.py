from scanpy.tools import pca
from .correctors import FWERCorrector

class Embedder:
    """
    Base class for Embedders. This class serves as an "identity" embedder, returning the input data unchanged.
    """
    def __init__(self):
        """
        Initializes the Embedder. This constructor is intended to be extended by child classes.
        """
        pass

    def embed(self, X):
        """
        Returns the input data unchanged.

        Parameters
        ----------
        X : numpy.ndarray or scipy.spmatrix
            The input data to be embedded.

        Returns
        -------
        X : numpy.ndarray or scipy.spmatrix
            The unchanged input data.
        """
        return X

class SVDEmbedder(Embedder):
    """
    Embedder that projects data to its top right singular vectors.
    """

    def __init__(self, n_comps=50, svd_mat=None, **kwargs):
        """
        Constructor

        Parameters
        ----------
        n_comps : int, optional
            Number of top singular vectors to keep. Defines output dimensionality. Default is 50.
        svd_mat : numpy.ndarray or scipy.spmatrix, optional
            If provided, project the input to the right singular vectors of this matrix. If None (default), project to singular vectors of the input.
        kwargs : dict, optional
            Additional arguments passed to scanpy.tl.pca.
        """
        self.n_comps = n_comps
        self.kwargs = kwargs
        self.svd_mat = svd_mat

    def embed(self, X, keep_loadings=False):
        """
        Project X onto a set of top right singular vectors, and return the result.

        Parameters
        ----------
        X : numpy.ndarray or scipy.spmatrix
            Data to be reduced.
        keep_loadings : bool, optional
            Whether to keep the loading vectors. If True, the loading vectors are stored as self.loadings. Default is False.

        Returns
        -------
        numpy.ndarray or scipy.spmatrix
            A low-dimensional representation of X.
        """
        if self.svd_mat is None:
            self.svd_mat = X  # Use the input matrix if no SVD matrix is provided

        # Perform PCA to get the top right singular vectors
        a, bt, c, d = pca(self.svd_mat, n_comps=self.n_comps, return_info=True, **self.kwargs)

        if keep_loadings:
            self.loadings = bt.transpose()  # Store the loading vectors if requested

        return X @ bt.transpose()  # Project the input data onto the top right singular vectors

class SCAEmbedder(Embedder):
    """
    Embedder that performs SCA on the input.
    """

    def __init__(self, scorer, connector=None, n_comps=50, iters=1):
        """
        Constructor.

        Parameters
        ----------
        n_comps : int
            Number of Shannon components to compute.
        scorer : Scorer
            Scorer object used to generate scores from neighborhoods.
        connector : Connector or None, optional
            Connector object used to generate neighborhoods of the input data. If None, you must provide neighborhoods when running ``embed``.
        iters : int, optional
            Number of SCA iterations to run. Default is 1.
        """
        self.n_comps = n_comps
        self.iters = iters
        self.scorer = scorer
        self.connector = connector
        self.embedding_dict = {}

    def embed(self, X, keep_scores=False, keep_loadings=False, keep_all_iters=False, nbhds=None, verbose=True):
        """
        Compute the SCA embedding.

        Parameters
        ----------
        X : np.ndarray or scipy.spmatrix
            Input data, with one row per observation and one column per feature.
        keep_scores : bool, optional
            If True, SCA scores are retained in the .scores attribute of this object. Default is False.
        keep_loadings : bool, optional
            If True, Shannon Component loadings are retained in the .loadings attribute. Default is False.
        keep_all_iters : bool, optional
            If True, intermediate embeddings after each iteration are retained in self.embedding_dict, a dictionary of embeddings keyed by iteration number. Default is False.
        nbhds : list or None, optional
            Neighborhoods to use when computing scores. Overrides the Connector, if specified at construction. If Connector=None at construction, this parameter is not optional.
        verbose : bool, optional
            If True, print progress during execution. Default is True.

        Returns
        -------
        np.ndarray or scipy.spmatrix
            The SCA embedding of the input data.
        """
        assert (self.connector is not None) or (nbhds is not None), "must specify connector or neighborhoods"

        if self.iters == 0:
            # Base case: just do PCA
            embedder = SVDEmbedder(n_comps=self.n_comps)

            if keep_all_iters:
                self.embedding_dict = {}

            return embedder.embed(X)
        else:
            # Make neighborhoods if not provided
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

            # Compute information scores
            scores = self.scorer.score(X, nbhds)

            # Perform SVD on the scores to get the final embedding
            final_embedder = SVDEmbedder(n_comps=self.n_comps, svd_mat=scores)
            result = final_embedder.embed(X, keep_loadings=True)

            if keep_scores:
                self.scores = scores
            if keep_loadings:
                self.loadings = final_embedder.loadings
            if keep_all_iters:
                self.embedding_dict[self.iters] = result

            return result




