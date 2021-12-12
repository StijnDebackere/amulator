import warnings

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class PoissonLikelihood(_OneDimensionalLikelihood):
    r"""A Poisson likelihood/noise model for GP regression for
    Poisson-like data :math:`N` with possible super-Poisson errors
    with variance :math:`\alpha N`, where :math:`\alpha > 1` is the
    super_poisson_ratio.

    :param bool log: model predicts log counts :math:`\log N`
    :param bool ratio: model predicts ratio to mean :math:`N / \maths N \rangle`
    """
    def __init__(self, log=True, mean=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = log
        self.mean = mean

    def _get_kwargs(self, model_samples, **kwargs):
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

        # poisson rate is set by function_samples
        rate = model_samples
        rate = rate.clamp(1e-2)
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
