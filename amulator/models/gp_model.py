import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class GPModel(ApproximateGP):
    def __init__(
            self,
            inducing_points,
            mean_prior=None,
            outputscale_prior=None,
            lengthscale_prior=None,
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0),
            dtype=inducing_points.dtype,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(prior=mean_prior)
        self.covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=inducing_points.shape[1],
                lengthscale_prior=lengthscale_prior,
            ),
            outputscale_prior=outputscale_prior,
        )
        if mean_prior is not None:
            self.mean_module.sample_from_prior("mean_prior")
        if outputscale_prior is not None:
            self.covar_module.sample_from_prior("outputscale_prior")
        if lengthscale_prior is not None:
            self.covar_module.base_kernel.sample_from_prior("lengthscale_prior")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
