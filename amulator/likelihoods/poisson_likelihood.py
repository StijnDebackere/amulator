import warnings

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class PoissonLikelihood(_OneDimensionalLikelihood):
    r"""A Poisson likelihood/noise model for GP regression for
    Poisson-like data :math:`N`.

    :param bool n2N: model predicts number density :math:`n = N / n2N` instead of :math:`N`
    :param bool log: model predicts log counts :math:`\log N`
    :param bool mean: model predicts ratio to mean :math:`N / \maths N \rangle`
    """
    def __init__(self, n2N=False, log=True, mean=True, likelihood_kwargs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n2N = n2N
        self.log = log
        self.mean = mean
        self.likelihood_kwargs = {} if likelihood_kwargs is None else likelihood_kwargs

    def _get_kwargs(self, model_samples, **kwargs):
        if self.log:
            model_samples = model_samples.exp()

        # model predicts ratio to mean
        if self.mean:
            if "model_mean" not in kwargs:
                raise ValueError("model_mean needs to be specified if {self.mean=}")

            model_mean = kwargs["model_mean"]
            if self.log:
                model_mean = model_mean.exp()

            if torch.any(model_mean == 0.):
                warnings.warn("model_mean contains 0.")

            # take out ratio
            model_samples = model_samples * model_mean

        # convert to actual counts if normalized
        if self.n2N:
            if "n2N" not in kwargs:
                raise ValueError("'n2N' needs to be in kwargs specified if {self.n2N=}")

            n2N = kwargs["n2N"]
            model_samples = model_samples * n2N

        # poisson rate is set by function_samples
        rate = model_samples
        rate = rate.clamp(1e-2)
        return {
            "rate": rate,
        }

    def forward(self, log_function_samples, **kwargs):
        poisson_kwargs = self._get_kwargs(log_function_samples, **kwargs)
        return base_distributions.Poisson(**poisson_kwargs, **self.likelihood_kwargs)

    def log_marginal(self, observations, log_function_dist, *args, **kwargs):
        marginal = self.marginal(log_function_dist, *args, **kwargs)
        return marginal.log_prob(observations.to(torch.int))

    def marginal(self, log_function_dist, **kwargs):
        poisson_kwargs = self._get_kwargs(log_function_dist.mean, **kwargs)
        return base_distributions.Poisson(**poisson_kwargs, **self.likelihood_kwargs)

    def expected_log_prob(self, observations, log_function_dist, *args, **kwargs):
        prob_lambda = lambda log_function_samples: self.forward(
            log_function_samples, *args, **kwargs
        ).log_prob(observations.to(torch.int))
        log_prob = self.quadrature(prob_lambda, log_function_dist)
        return log_prob
