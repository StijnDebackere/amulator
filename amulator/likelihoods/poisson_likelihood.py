from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class PoissonLikelihood(_OneDimensionalLikelihood):
    r"""A Poisson likelihood/noise model for GP regression for
    Poisson-like data.

    Model predicts :math:`\log_{10} N` and data is :math:`N`.

    """
    def _get_kwargs(self, log_function_samples, **kwargs):
        function_samples = np.exp(log_function_samples)

        # poisson rate is set by function_samples
        rate = function_samples
        return {
            "rate": rate,
        }

    def forward(self, log_function_samples, **kwargs):
        poisson_kwargs = self._get_kwargs(log_function_samples, **kwargs)
        return base_distributions.Poisson(**poisson_kwargs)

    def log_marginal(self, observations, log_function_dist, *args, **kwargs):
        marginal = self.marginal(log_function_dist, *args, **kwargs)
        return marginal.log_prob(observations.to(torch.int))

    def marginal(self, log_function_dist, **kwargs):
        poisson_kwargs = self._get_kwargs(log_function_dist.mean, **kwargs)
        return base_distributions.Poisson(**poisson_kwargs)

    def expected_log_prob(self, observations, log_function_dist, *args, **kwargs):
        prob_lambda = lambda log_function_samples: self.forward(
            log_function_samples, *args, **kwargs
        ).log_prob(observations.to(torch.int))
        log_prob = self.quadrature(prob_lambda, log_function_dist)
        return log_prob
