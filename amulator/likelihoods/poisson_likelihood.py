from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class PoissonLikelihood(_OneDimensionalLikelihood):
    r"""A Poisson likelihood/noise model for GP regression for
    Poisson-like data.

    Data follows Poisson distribution with :math:`rate`.

    """
    def _get_kwargs(self, function_samples, **kwargs):
        function_samples = 10 ** function_samples

        # poisson rate is set by function_samples
        rate = function_samples
        return {
            "rate": rate,
        }

    def forward(self, function_samples, **kwargs):
        poisson_kwargs = self._get_kwargs(function_samples, **kwargs)
        return base_distributions.Poisson(**poisson_kwargs)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        marginal = self.marginal(function_dist, *args, **kwargs)
        return marginal.log_prob(observations.to(torch.int))

    def marginal(self, function_dist, **kwargs):
        poisson_kwargs = self._get_kwargs(function_dist.mean, **kwargs)
        return base_distributions.Poisson(**poisson_kwargs)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        prob_lambda = lambda function_samples: self.forward(
            function_samples, *args, **kwargs
        ).log_prob(observations.to(torch.int))
        log_prob = self.quadrature(prob_lambda, function_dist)
        return log_prob
