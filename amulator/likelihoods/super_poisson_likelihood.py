import warnings

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class SuperPoissonLikelihoodBase(_OneDimensionalLikelihood):
    r"""A NegativeBinomial likelihood/noise model for GP regression for
    Poisson-like data :math:`N` with possible super-Poisson errors
    with variance :math:`\alpha N`, where :math:`\alpha > 1` is the
    poisson_ratio.

    :param bool n2N: model predicts number density :math:`n = N / n2N` instead of :math:`N`
    :param bool log: model predicts log :math:`\log N`
    :param bool mean: model predicts ratio to mean :math:`N / \langle N \rangle`
    """
    def __init__(self, likelihood_kwargs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.likelihood_kwargs = {} if likelihood_kwargs is None else likelihood_kwargs

    def _get_kwargs(self, model_samples, *args, **kwargs):
        poisson_ratio = kwargs.get("poisson_ratio", None)

        if poisson_ratio is None:
            raise ValueError("poisson_ratio should be in **kwargs")

        if (poisson_ratio < 1.).any():
            raise ValueError("poisson_ratio should be > 1 for SuperPoissonLikelihood")

        # ensure poisson_ratio > 1 to avoid divergences in r & probs
        alpha = poisson_ratio + 1e-6

        model_obs = self.transform_model(model_samples, *args, **kwargs)

        total_count = model_obs / (alpha - 1)
        probs = 1 - alpha ** (-1)

        # ensure small value to prevent NaN's in log_prob
        total_count[total_count == 0.] = 1e-6
        return {
            "total_count": total_count,
            "probs": probs,
        }

    def transform_model(self, model_samples, *args, **kwargs):
        """Convert the latent model_samples to the observations to compute the likelihood."""
        raise NotImplementedError

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


class SuperPoissonLikelihoodMeanStd(SuperPoissonLikelihoodBase):
    r"""A NegativeBinomial likelihood/noise model for GP regression for
    Poisson-like data :math:`N` with possible super-Poisson errors
    with variance :math:`\alpha N`, where :math:`\alpha > 1` is the
    poisson_ratio.

    :param bool n2N: model predicts number density :math:`n = N / n2N` instead of :math:`N`
    :param bool log: model predicts log :math:`\log N`
    :param bool mean: model predicts ratio to mean :math:`N / \langle N \rangle`
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

        # N = n * n2N, f = log(n) - log(mean_n) / sigma(log(n / mean_n))
        obs = (model_samples * model_sigma + model_mean).exp() * model_to_obs
        return obs
