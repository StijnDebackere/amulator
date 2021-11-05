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
):
    """Load the model_trainer from the checkpoint file with state_dict for
    the model, likelihood, mll and optimizer."""
    checkpoint_info = torch.load(checkpoint_file)

    inducing_points = checkpoint_info["model"]["variational_strategy.inducing_points"]
    model = GPModel(inducing_points)
    model.load_state_dict(checkpoint_info["model"])
    likelihood = SuperPoissonLikelihood()
    likelihood.load_state_dict(checkpoint_info["likelihood"])
    mll = VariationalELBO(likelihood, model, num_data=len(dataloader.dataset))
    optimizer.load_state_dict(checkpoint_info["optimizer"])

    return GPModelTrainer(
        model=model,
        likelihood=likelihood,
        mll=mll,
        optimizer=optimizer,
    )
