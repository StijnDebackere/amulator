import warnings

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class PoissonLikelihoodBase(_OneDimensionalLikelihood):
    r"""A Poisson likelihood/noise model for GP regression for
    Poisson-like data :math:`N`.
    """
    def __init__(self, likelihood_kwargs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.likelihood_kwargs = {} if likelihood_kwargs is None else likelihood_kwargs

    def _get_kwargs(self, model_samples, **kwargs):
        rate = self.transform_model(model_samples)
        rate = rate.clamp(1e-2)
        return {
            "rate": rate,
        }

    def forward(self, model_samples, **kwargs):
        poisson_kwargs = self._get_kwargs(model_samples, **kwargs)
        return base_distributions.Poisson(**poisson_kwargs, **self.likelihood_kwargs)

    def log_marginal(self, observations, model_dist, *args, **kwargs):
        marginal = self.marginal(model_dist, *args, **kwargs)
        return marginal.log_prob(observations.to(torch.int))

    def marginal(self, model_dist, **kwargs):
        poisson_kwargs = self._get_kwargs(model_dist.mean, **kwargs)
        return base_distributions.Poisson(**poisson_kwargs, **self.likelihood_kwargs)

    def expected_log_prob(self, observations, model_dist, *args, **kwargs):
        prob_lambda = lambda model_samples: self.forward(
            model_samples, *args, **kwargs
        ).log_prob(observations.to(torch.int))
        log_prob = self.quadrature(prob_lambda, model_dist)
        return log_prob


class PoissonLikelihoodMeanStd(PoissonLikelihoodBase):
    r"""A Poisson likelihood/noise model for GP regression for
    Poisson-like data :math:`N` with latent model :math:`f = (N - mean(N)) / sigma(N)`
    """
    def __init__(self, likelihood_kwargs=None, *args, **kwargs):
        super().__init__(likelihood_kwargs=likelihood_kwargs, *args, **kwargs)

    def transform_model(self, model_samples, *args, **kwargs):
        """Convert the latent model_samples to the observations to compute the likelihood."""
        model_mean = kwargs.get("model_mean", None)
        model_sigma = kwargs.get("model_sigma", None)
        if model_mean is None or model_sigma is None:
            raise ValueError("model_mean and model_sigma should be in **kwargs")

        model_to_obs = kwargs.get("model_to_obs", None)
        if model_to_obs is None:
            raise ValueError("model_to_obs should be in **kwargs")

        # N = n * n2N, f = (n - mean_n) / sigma_n
        obs = (model_samples * model_sigma + model_mean) * model_to_obs
        return obs
