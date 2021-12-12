import math
import warnings

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class GaussianLikelihood(_OneDimensionalLikelihood):
    r"""A Normal likelihood/noise model for GP regression for data
    :math:`N` with noise.

    :param bool log: model predicts log counts :math:`\log N`
    :param bool ratio: model predicts ratio to mean :math:`N / \maths N \rangle`

    """
    def __init__(self, log=True, mean=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = log
        self.mean = mean

    def _get_kwargs(self, model_samples, **kwargs):
        if "noise" not in kwargs:
            raise ValueError("'noise' should be in kwargs")

        noise = kwargs["noise"]
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

        loc = model_samples
        scale = noise.clamp(1e-2).sqrt()
        return {
            "loc": loc,
            "scale": scale,
        }

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
