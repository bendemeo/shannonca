import torch


def log_likelihood(nbhd_means, feature_means, feature_vars, k, past_comps = []):
    # given neighborhood expression data, construct likelihood function
    # should work in pytorch
    n_samples = nbhd_means.shape[0]

    nbhd_means = torch.tensor(nbhd_means).double()
    feature_means = torch.tensor(feature_means).double()
    feature_vars = torch.tensor(feature_vars).double()

    def f(coefs):
        coefs_orth = coefs
        for i,comp in enumerate(past_comps):
            #project orthogonally
            coefs_orth = coefs_orth - torch.dot(coefs_orth, comp) * comp

        observed_means = torch.matmul(nbhd_means, torch.reshape(coefs_orth, (-1, 1)))

        # mean and variance of metagene defined by coef, under
        # independent gene hypothesis
        # variance gets scaled down due to averaging.
        theoretical_mean = torch.dot(feature_means, coefs_orth)
        theoretical_var = torch.div(torch.dot(torch.pow(coefs_orth, 2), feature_vars), float(k))

        #         print(theoretical_mean)
        #         print(theoretical_var)

        result = (-1 * n_samples / 2.) * torch.log(theoretical_var) - torch.div(
            torch.sum(torch.pow(observed_means - theoretical_mean, 2)),
            2 * theoretical_var)
        return (result)

    return (f)
