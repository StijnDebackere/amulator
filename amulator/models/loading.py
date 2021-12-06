from gpytorch.mlls import VariationalELBO
from ignite.handlers import Checkpoint
import torch

from amulator.likelihoods import SuperPoissonLikelihood
from amulator.models import GPModel, ExactGPModel
from amulator.training.trainer import GPModelTrainer


def read_super_poisson_gp_model_trainer(
        checkpoint_file,
        dataloader,
        optimizer,
        log=True,
        mean=True,
):
    """Load the model_trainer from the checkpoint file with state_dict for
    the model, likelihood, mll and optimizer."""
    checkpoint_info = torch.load(checkpoint_file)

    inducing_points = checkpoint_info["model"]["variational_strategy.inducing_points"]
    model = GPModel(inducing_points)
    model.load_state_dict(checkpoint_info["model"])
    likelihood = SuperPoissonLikelihood(log=log, mean=mean)
    likelihood.load_state_dict(checkpoint_info["likelihood"])
    mll = VariationalELBO(likelihood, model, num_data=len(dataloader.dataset))
    optimizer = optimizer(model.parameters())
    optimizer.load_state_dict(checkpoint_info["optimizer"])

    criterion_extra_keys = ["super_poisson_ratio"]
    if mean:
        criterion_extra_keys.append("model_mean")

    return GPModelTrainer(
        model=model,
        likelihood=likelihood,
        mll=mll,
        optimizer=optimizer,
        criterion_extra_keys=criterion_extra_keys,
    )
