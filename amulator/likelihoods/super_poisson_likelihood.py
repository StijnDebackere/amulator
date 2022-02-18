import warnings

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class SuperPoissonLikelihood(_OneDimensionalLikelihood):
    r"""A NegativeBinomial likelihood/noise model for GP regression for
    Poisson-like data :math:`N` with possible super-Poisson errors
    with variance :math:`\alpha N`, where :math:`\alpha > 1` is the
    poisson_ratio.

    :param bool n2N: model predicts number density :math:`n = N / n2N` instead of :math:`N`
    :param bool log: model predicts log :math:`\log N`
    :param bool mean: model predicts ratio to mean :math:`N / \langle N \rangle`
    """
    def __init__(self, n2N=False, log=True, mean=True, likelihood_kwargs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n2N = n2N
        self.log = log
        self.mean = mean
        self.likelihood_kwargs = {} if likelihood_kwargs is None else likelihood_kwargs

    def _get_kwargs(self, model_samples, **kwargs):
        """convert model_samples to correct counts"""
        if "poisson_ratio" not in kwargs:
            raise ValueError("'poisson_ratio' should be in kwargs")

        poisson_ratio = kwargs["poisson_ratio"]
        # ensure noise is always super-poisson
        if torch.isnan(poisson_ratio).any() or (poisson_ratio < 1.).any():
            raise ValueError("poisson_ratio contains NaNs or values < 1.")
            # warnings.warn("forcing nans in poisson_ratio to 1")
            # poisson_ratio[poisson_ratio.isnan()] = 1.

        if self.log:
            model_samples = model_samples.exp()

        # model predicts ratio to mean
        if self.mean:
            if "model_mean" not in kwargs:
                raise ValueError("model_mean needs to be specified if {self.mean=}")

            model_mean = kwargs["model_mean"]
            if self.log:
                model_mean = model_mean.exp()

            if torch.any(model_mean == 0.):
                warnings.warn("model_mean contains 0.")

            # take out ratio
            model_samples = model_samples * model_mean

        # convert to actual counts if normalized
        if self.n2N:
            if "n2N" not in kwargs:
                raise ValueError("'n2N' needs to be in kwargs specified if {self.n2N=}")

            n2N = kwargs["n2N"]
            model_samples = model_samples * n2N

        # ensure poisson_ratio > 1 to avoid divergences in r & probs
        alpha = poisson_ratio + 1e-6

        # total_count := total number of failures (r)
        # probs := success probability in individual Bernoulli trials
        # mean = pr / (1 - p)
        # var = mean / (1 - p) = poisson_ratio * mean
        # => poisson_ratio = 1 / (1 - p)
        # => p / (1 - p) = poisson_ratio - 1
        r = model_samples / (alpha - 1)
        probs = 1 - alpha ** (-1)

        # ensure small value to prevent NaN's in log_prob
        r[r == 0.] = 1e-6
        return {
            "total_count": r,
            "probs": probs,
        }

    def forward(self, model_samples, **kwargs):
        binom_kwargs = self._get_kwargs(model_samples, **kwargs)
        return base_distributions.NegativeBinomial(**binom_kwargs, **self.likelihood_kwargs)

    def log_marginal(self, observations, model_dist, *args, **kwargs):
        marginal = self.marginal(model_dist, *args, **kwargs)
        return marginal.log_prob(observations.to(torch.int))

    def marginal(self, model_dist, **kwargs):
        binom_kwargs = self._get_kwargs(model_dist.mean, **kwargs)
        return base_distributions.NegativeBinomial(**binom_kwargs, **self.likelihood_kwargs)

    def expected_log_prob(self, observations, model_dist, *args, **kwargs):
        prob_lambda = lambda model_samples: self.forward(
            model_samples, *args, **kwargs
        ).log_prob(observations.to(torch.int))
        log_prob = self.quadrature(prob_lambda, model_dist)
        return log_prob
