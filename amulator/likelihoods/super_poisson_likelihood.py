import warnings

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class SuperPoissonLikelihood(_OneDimensionalLikelihood):
    r"""A NegativeBinomial likelihood/noise model for GP regression for
    Poisson-like data :math:`N` with possible super-Poisson errors
    with variance :math:`\alpha N`, where :math:`\alpha > 1` is the
    poisson_ratio.

    :param bool log: model predicts log counts :math:`\log N`
    :param bool ratio: model predicts ratio to mean :math:`N / \maths N \rangle`
    """
    def __init__(self, log=True, mean=True, likelihood_kwargs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = log
        self.mean = mean
        self.likelihood_kwargs = {} if likelihood_kwargs is None else likelihood_kwargs

    def _get_kwargs(self, model_samples, **kwargs):
        if "poisson_ratio" not in kwargs:
            raise ValueError("'poisson_ratio' should be in kwargs")

        poisson_ratio = kwargs["poisson_ratio"]
        # ensure noise is always super-poisson
        if torch.isnan(poisson_ratio).any():
            warnings.warn("forcing nans in poisson_ratio to 1")
            poisson_ratio[poisson_ratio.isnan()] = 1.

        if (poisson_ratio < 1).any():
            warnings.warn("forcing poisson_ratio >= 1")
            poisson_ratio[poisson_ratio < 1.] = 1.

        if self.log:
            model_samples = model_samples.exp()

        if self.mean:
            if "model_mean" not in kwargs:
                raise ValueError("model_mean needs to be specified if {self.mean=}")

            else:
                model_mean = kwargs["model_mean"]
                if self.log:
                    model_mean = model_mean.exp()

                if torch.any(model_mean == 0.):
                    warnings.warn("model_mean contains 0.")

            model_samples = model_samples * model_mean

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
