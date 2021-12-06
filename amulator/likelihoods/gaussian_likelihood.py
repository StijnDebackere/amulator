import warnings

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
import torch


class GaussianLikelihood(gaussian_likelihood.FixedNoiseGaussianLikelihood):
    r"""A Likelihood that assumes fixed heteroscedastic noise. This is useful when you have fixed, known observation
    noise for each training example.

    Note that this likelihood takes an additional argument when you call it, `noise`, that adds a specified amount
    of noise to the passed MultivariateNormal. This allows for adding known observational noise to test data.

    .. note::
        This likelihood can be used for exact or approximate inference.

    :param noise: Known observation noise (variance) for each training example.
    :type noise: torch.Tensor (... x N)
    :param learn_additional_noise: Set to true if you additionally want to
        learn added diagonal noise, similar to GaussianLikelihood.
    :type learn_additional_noise: bool, optional
    :param batch_shape: The batch shape of the learned noise parameter (default
        []) if :obj:`learn_additional_noise=True`.
    :type batch_shape: torch.Size, optional

    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)

    Example:
        >>> train_x = torch.randn(55, 2)
        >>> noises = torch.ones(55) * 0.01
        >>> likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
        >>> pred_y = likelihood(gp_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 2)
        >>> test_noises = torch.ones(21) * 0.02
        >>> pred_y = likelihood(gp_model(test_x), noise=test_noises)
    """
    def __init__(
        self,
        noise: Tensor,
        learn_additional_noise: Optional[bool] = False,
        log: bool = True,
        mean: bool = True,
        batch_shape: Optional[torch.Size] = torch.Size(),
        **kwargs: Any,
    ) -> None:
        # initialize FixedNoisegaussianlikelihood, as usual
        # we will overwrite the forward methods to take into account that model = log10(observations / mean)
        super().__init__(
            noise=noise,
            learn_additional_noise=learn_additional_noise,
            batch_shape=batch_shape,
            **kwargs,
        )
        self.log = log
        self.mean = mean

    def expected_log_prob(self, target, dist, *params, **kwargs):
        mean, variance = dist.mean, dist.variance
        # dist gives ratio wrt model_mean, convert to target
        if self.mean:
            if "model_mean" not in kwargs:
                raise ValueError("model_mean should be in kwargs")

            model_mean = kwargs["model_mean"]
            if log:
                model_mean = model_mean.exp()

            mean = mean * model_mean
            variance = variance * model_mean ** 2

        # if dist models log(target), predictions follow lognormal distribution
        if self.log:
            lognorm = base_distributions.LogNormal(loc=mean, scale=variance.sqrt())
            mean, variance = lognorm.mean, lognorm.variance

        noise = self._shaped_noise_covar(mean.shape, *params, **kwargs).diag()
        res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)
        return res

    def forward(self, model_samples, *params, **kwargs):
        if self.log:
            model_samples = model_samples.exp()

        if self.mean:
            if "model_mean" not in kwargs:
                raise ValueError("model_mean should be in kwargs")

            model_mean = kwargs["model_mean"]
            if log:
                model_mean = model_mean.exp()

            model_samples = model_mean * model_samples

        noise = self._shaped_noise_covar(model_samples.shape, *params, **kwargs).diag()
        return base_distributions.Normal(model_samples, noise.sqrt())

    def marginal(self, dist, *params, **kwargs):
        # dist gives ratio wrt model_mean, convert to observations
        if self.mean:
            if "model_mean" not in kwargs:
                raise ValueError("model_mean should be in kwargs")

            model_mean = kwargs["model_mean"]
            if log:
                model_mean = model_mean.exp()

            # X ~ N(a, b^2) => cX ~ N(ca, c^2b^2)
            mean = dist.mean * model_mean
            variance = dist.variance * model_mean ** 2
        else:
            mean, variance = dist.mean, dist.variance

        # if dist models log(target), predictions follow lognormal distribution
        if self.log:
            lognorm = base_distributions.LogNormal(loc=mean, scale=variance.sqrt())
            mean, variance = lognorm.mean, lognorm.variance

        mean, covar = dist.mean, dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return dist.__class__(mean, full_covar)
