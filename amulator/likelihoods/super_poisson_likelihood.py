from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class SuperPoissonLikelihood(_OneDimensionalLikelihood):
    r"""A NegativeBinomial likelihood/noise model for GP regression for
    Poisson-like data with possible super-Poisson errors.

    Model predicts :math:`\log_{10} f - \mathbb{E} \log_{10} f` and
    data is :math:`f` following a NegativeBinomial distribution with
    mean :math:`f` and variance :math:`\alpha f`, where :math:`\alpha > 1`
    is the super_poisson_ratio.

    """
    def _get_kwargs(self, log10_function_ratio_samples, **kwargs):
        if "super_poisson_ratio" not in kwargs:
            raise ValueError("'super_poisson_ratio' should be in kwargs")

        if "log10_function_mean" not in kwargs:
            raise ValueError("'log10_function_mean' should be in kwargs")

        super_poisson_ratio = kwargs["super_poisson_ratio"]
        # ensure noise is always super-poisson
        if torch.isnan(super_poisson_ratio).any() or (super_poisson_ratio < 1).any():
            raise ValueError("super_poisson_ratio contains NaNs and/or values < 1.")

        log10_function_mean = kwargs["log10_function_mean"]
        function_samples = 10 ** (log10_function_ratio_samples + log10_function_mean)

        # poisson noise is set by function_samples
        # ensure noise always > function_samples
        noise = function_samples * (super_poisson_ratio + 1e-2)

        r = function_samples ** 2 / (noise - function_samples)
        probs = function_samples / (r + function_samples)

        return {
            "total_count": r,
            "probs": probs,
        }

    def forward(self, log10_function_ratio_samples, **kwargs):
        binom_kwargs = self._get_kwargs(log10_function_ratio_samples, **kwargs)
        return base_distributions.NegativeBinomial(**binom_kwargs)

    def log_marginal(self, observations, log10_function_ratio_dist, *args, **kwargs):
        marginal = self.marginal(log10_function_ratio_dist, *args, **kwargs)
        return marginal.log_prob(observations.to(torch.int))

    def marginal(self, log10_function_ratio_dist, **kwargs):
        binom_kwargs = self._get_kwargs(log10_function_ratio_dist.mean, **kwargs)
        return base_distributions.NegativeBinomial(**binom_kwargs)

    def expected_log_prob(self, observations, log10_function_ratio_dist, *args, **kwargs):
        prob_lambda = lambda log10_function_ratio_samples: self.forward(
            log10_function_ratio_samples, *args, **kwargs
        ).log_prob(observations.to(torch.int))
        log_prob = self.quadrature(prob_lambda, log10_function_ratio_dist)
        return log_prob
