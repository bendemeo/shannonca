import unittest
from shannonca.generate import pancakes
from shannonca.scorers import  *
from shannonca.correctors import FWERCorrector
from sklearn.neighbors import NearestNeighbors
from shannonca.dimred import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from shannonca.score import info_score

class test_scorers(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(test_scorers, self).__init__(*args, **kwargs)
        self.testData = pancakes(n_pts=100, n_cor=10, n_anticor=10, n_noise=100).X
        self.seed = 20
        nn = NearestNeighbors(metric='cosine', n_neighbors=10); nn.fit(self.testData)
        self.nbhds = nn.kneighbors(return_distance=False)

    def test_WilcoxonScorer(self):
        scorer = WilcoxonScorer(seed=self.seed, verbose=True, corrector=FWERCorrector(n_tests=10))
        scores = scorer.score(self.testData, self.nbhds)

        old_scores = info_score(self.testData, nbhds=self.nbhds, n_tests=10, model='wilcoxon',
                                chunk_size=10000)

        self.assertTrue(np.max(scores - old_scores) <= 1e-10)

        sns.heatmap(scores.todense(), cmap='coolwarm')
        plt.show()
        plt.savefig('figs/wilcoxon_pancake_scores.png')

    def test_BinomialScorer(self):
        scorer = BinomialScorer(seed=self.seed, verbose=True, corrector=FWERCorrector(n_tests=10))
        scores = scorer.score(self.testData, self.nbhds)
        scores_old = info_score(self.testData, self.nbhds, chunk_size=1000000, n_tests=10, model='binomial')
        self.assertTrue(np.max(scores - scores_old) <= 1e-10)

    def test_Tscorer(self):
        scorer = TScorer(seed=self.seed, verbose=True, corrector=FWERCorrector(n_tests=10))
        scores = scorer.score(self.testData, self.nbhds)
        scores_old = info_score(self.testData, self.nbhds, chunk_size=1000000, n_tests=10, model='ttest')
        self.assertTrue(np.max(scores-scores_old) <= 1e-10)

    def test_ChunkedScorer(self):
        base_scorer = WilcoxonScorer(seed=self.seed, verbose=True, corrector=FWERCorrector(n_tests=10))
        chunked_scorer_1 = ChunkedScorer(base_scorer=base_scorer, chunk_size=10)
        chunked_scorer_2 = ChunkedScorer(base_scorer=base_scorer, max_size=500)


        unchunked_scores = base_scorer.score(self.testData, self.nbhds).todense()
        fixedchunk_scores = chunked_scorer_1.score(self.testData, self.nbhds).todense()
        maxchunk_scores = chunked_scorer_2.score(self.testData, self.nbhds).todense()

        self.assertTrue((unchunked_scores==fixedchunk_scores).all())
        self.assertTrue((fixedchunk_scores==unchunked_scores).all())

    def test_Tf_idfScorer(self):
        scorer = Tf_idfScorer()
        scores = scorer.score(self.testData).todense()
        sns.heatmap(scores)






