import math
import warnings

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class GaussianLikelihoodBase(_OneDimensionalLikelihood):
    r"""A Normal likelihood/noise model for GP regression for data with noise.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_kwargs(self, model_samples, *args, **kwargs):
        noise = kwargs.get("noise", None)
        if noise is None:
            raise ValueError("noise should be in kwargs")

        obs = self.transform_model(model_samples, *args, **kwargs)

        loc = obs
        scale = noise.clamp(1e-2).sqrt()
        return {
            "loc": loc,
            "scale": scale,
        }

    def transform_model(self, model_samples, *args, **kwargs):
        """Convert the latent model_samples to the observations to compute the likelihood."""
        raise NotImplementedError

    def forward(self, model_samples, **kwargs):
        normal_kwargs = self._get_kwargs(model_samples, **kwargs)
        return base_distributions.Normal(**normal_kwargs)

    def log_marginal(self, observations, model_dist, *args, **kwargs):
        marginal = self.marginal(model_dist, *args, **kwargs)
        return marginal.log_prob(observations.to(torch.int))

    def marginal(self, model_dist, **kwargs):
        normal_kwargs = self._get_kwargs(model_dist.mean, **kwargs)
        return base_distributions.Normal(**normal_kwargs)

    def expected_log_prob(self, observations, model_dist, *args, **kwargs):
        prob_lambda = lambda model_samples: self.forward(
            model_samples, *args, **kwargs
        ).log_prob(observations.to(torch.int))
        log_prob = self.quadrature(prob_lambda, model_dist)
        return log_prob


class GaussianLikelihoodMeanStd(GaussianLikelihoodBase):
    r"""A Normal likelihood/noise model for GP regression for data
    :math:`N` with noise.

    :param bool log: model predicts log counts :math:`\log N`
    :param bool ratio: model predicts ratio to mean :math:`N / \maths N \rangle`

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform_model(model_samples, *args, **kwargs):
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
