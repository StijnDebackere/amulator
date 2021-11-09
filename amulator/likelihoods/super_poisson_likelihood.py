from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class SuperPoissonLikelihood(_OneDimensionalLikelihood):
    r"""A NegativeBinomial likelihood/noise model for GP regression for
    Poisson-like data with possible supper-Poisson errors.

    Data is :math:`f` following a NegativeBinomial distribution with
    mean :math:`f` and variance :math:`\alpha f`, where :math:`\alpha`
    is the super_poisson_ratio.

    """
    def _get_kwargs(self, function_samples, **kwargs):
        if "super_poisson_ratio" not in kwargs:
            raise ValueError("'super_poisson_ratio' should be in kwargs")

        super_poisson_ratio = kwargs["super_poisson_ratio"]
        # ensure noise is always super-poisson
        if torch.isnan(super_poisson_ratio).any() or (super_poisson_ratio < 1).any():
            raise ValueError("super_poisson_ratio contains NaNs and/or values < 1.")

        # poisson noise is set by function_samples
        # ensure noise always > function_samples
        noise = function_samples * (super_poisson_ratio + 1e-2)

        r = function_samples ** 2 / (noise - function_samples)
        probs = function_samples / (r + function_samples)

        return {
            "total_count": r,
            "probs": probs,
        }

    def forward(self, function_samples, **kwargs):
        binom_kwargs = self._get_kwargs(function_samples, **kwargs)
        return base_distributions.NegativeBinomial(**binom_kwargs)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        marginal = self.marginal(function_dist, *args, **kwargs)
        return marginal.log_prob(observations.to(torch.int))

    def marginal(self, function_dist, **kwargs):
        binom_kwargs = self._get_kwargs(function_dist.mean, **kwargs)
        return base_distributions.NegativeBinomial(**binom_kwargs)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        prob_lambda = lambda function_samples: self.forward(
            function_samples, *args, **kwargs
        ).log_prob(observations.to(torch.int))
        log_prob = self.quadrature(prob_lambda, function_dist)
        return log_prob
