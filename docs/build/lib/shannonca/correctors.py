import numpy as np

def taylor_exp(x, n):
    # taylor series of 1-(1-x)^n
    # use for multiple test correction if x is very small.
    return (n * x
            - 0.5 * np.power(x, 2) * (n - 1) * n
            + (1. / 6.) * np.power(x, 3) * (n - 2) * (n - 1) * n
            - (1. / 24) * np.power(x, 4) * (n - 3) * (n - 2) * (n - 1) * n)



class FWERCorrector:
    def __init__(self, n_tests=1):
        self.n_tests = n_tests

    def correct(self, pvals):
        if self.n_tests > 1:
            # use FWER to correct for testing many genes
            pvals[pvals > 1e-10] = 1 - np.power(1 - pvals[pvals > 1e-10], self.n_tests)
            pvals[pvals <= 1e-10] = taylor_exp(pvals[pvals <= 1e-10], self.n_tests)
        return(pvals)