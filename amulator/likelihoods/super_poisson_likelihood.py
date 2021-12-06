import warnings

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class SuperPoissonLikelihood(_OneDimensionalLikelihood):
    r"""A NegativeBinomial likelihood/noise model for GP regression for
    Poisson-like data :math:`N` with possible super-Poisson errors
    with variance :math:`\alpha N`, where :math:`\alpha > 1` is the
    super_poisson_ratio.

    :param bool log: model predicts log counts :math:`\log N`
    :param bool ratio: model predicts ratio to mean :math:`N / \maths N \rangle`
    """
    def __init__(self, log=True, mean=True):
        super().__init__()
        self.log = log
        self.mean = mean

    def _get_kwargs(self, model_samples, **kwargs):
        if "super_poisson_ratio" not in kwargs:
            raise ValueError("'super_poisson_ratio' should be in kwargs")

        super_poisson_ratio = kwargs["super_poisson_ratio"]
        # ensure noise is always super-poisson
        if torch.isnan(super_poisson_ratio).any() or (super_poisson_ratio < 1).any():
            raise ValueError("super_poisson_ratio contains NaNs and/or values < 1.")

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

        # ensure super_poisson_ratio > 1 to avoid divergences in r & probs
        super_poisson_ratio_jitter = super_poisson_ratio + 1e-6

        alpha = super_poisson_ratio_jitter
        # total_count := total number of failures (r)
        # probs := success probability in individual Bernoulli trials
        # mean = pr / (1 - p)
        # var = mean / (1 - p) = super_poisson_ratio * mean
        # => super_poisson_ratio = 1 / (1 - p)
        # => p / (1 - p) = super_poisson_ratio - 1
        r = model_samples / (super_poisson_ratio_jitter - 1)
        probs = 1 - super_poisson_ratio_jitter ** (-1)

        return {
            "total_count": r,
            "probs": probs,
        }

    def forward(self, model_samples, **kwargs):
        binom_kwargs = self._get_kwargs(model_samples, **kwargs)
        return base_distributions.NegativeBinomial(**binom_kwargs)

    def log_marginal(self, observations, model_dist, *args, **kwargs):
        marginal = self.marginal(model_dist, *args, **kwargs)
        return marginal.log_prob(observations.to(torch.int))

    def marginal(self, model_dist, **kwargs):
        binom_kwargs = self._get_kwargs(model_dist.mean, **kwargs)
        return base_distributions.NegativeBinomial(**binom_kwargs)

    def expected_log_prob(self, observations, model_dist, *args, **kwargs):
        prob_lambda = lambda model_samples: self.forward(
            model_samples, *args, **kwargs
        ).log_prob(observations.to(torch.int))
        log_prob = self.quadrature(prob_lambda, model_dist)
        return log_prob
