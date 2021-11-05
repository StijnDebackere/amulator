import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
            self,
            train_x,
            train_y,
            likelihood,
            mean_prior=None,
            outputscale_prior=None,
            lengthscale_prior=None,
    ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(prior=mean_prior)
        self.covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=train_x.shape[1],
                lengthscale_prior=lengthscale_prior,
            )
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
