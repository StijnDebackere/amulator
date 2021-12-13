from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
from ignite.handlers import Checkpoint
import torch

from amulator.likelihoods import PoissonLikelihood, SuperPoissonLikelihood, GaussianLikelihood
from amulator.models import GPModel, ExactGPModel
from amulator.training.trainer import GPModelTrainer


TRAINER_OPTIONS = {
    "poisson": get_poisson_model_trainer,
    "gaussian": get_gaussian_model_trainer,
    "super_poisson": get_super_poisson_model_trainer,
}


def get_filename_prefix(save_prefix, model_trainer, save_suffix):
    if save_suffix is None:
        save_suffix = ""
    if save_prefix is None:
        save_prefix = ""

    model_name = type(model_trainer.model).__name__
    likelihood_name = type(model_trainer.likelihood).__name__
    optimizer_name = type(model_trainer.optimizer).__name__
    model_info = f"{model_name}_{likelihood_name}_optim_{optimizer_name}"
    filename_prefix = f"{save_prefix}_{model_info}_{save_suffix}".strip("_")

    return filename_prefix


def get_gaussian_model_trainer(
        dataloader,
        num_inducing,
        lr,
        learn_inducing_locations=True,
        model_kwargs=None,
        likelihood_kwargs=None,
        optimizer_kwargs=None,
):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    likelihood_kwargs = {} if likelihood_kwargs is None else likelihood_kwargs
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

    X = dataloader.dataset[:]["X"]
    yvar = dataloader.dataset[:]["yvar"]
    if learn_inducing_locations:
        # initialize GP model
        inducing_points = (
            torch.rand((num_inducing, X.shape[-1]))
            * (X.max(dim=0).values - X.min(dim=0).values)
        ) + X.min(dim=0).values
    else:
        inducing_points = (
            (X - X.min(dim=0).values) / (X.max(dim=0).values - X.min(dim=0).values)
        )
    model = GPModel(
        inducing_points,
        learn_inducing_locations=learn_inducing_locations,
        **model_kwargs,
    )
    likelihood = GaussianLikelihood(**likelihood_kwargs)
    mll = VariationalELBO(likelihood, model, num_data=len(dataloader.dataset))
    criterion_kwargs = {"noise": "yvar"}
    if likelihood.mean:
        criterion_kwargs["model_mean"] = "model_mean"

    # determine optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, **optimizer_kwargs)
    model_trainer = GPModelTrainer(
        model=model,
        likelihood=likelihood,
        mll=mll,
        optimizer=optimizer,
        criterion_kwargs=criterion_kwargs,
    )
    return model_trainer


def read_gaussian_gp_model_trainer(
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
    likelihood = GaussianLikelihood(log=log, mean=mean)
    likelihood.load_state_dict(checkpoint_info["likelihood"])
    mll = VariationalELBO(likelihood, model, num_data=len(dataloader.dataset))
    optimizer = optimizer(model.parameters())
    optimizer.load_state_dict(checkpoint_info["optimizer"])

    criterion_kwargs = {"noise": "yvar"}
    if mean:
        criterion_kwargs["model_mean"] = "model_mean"

    return GPModelTrainer(
        model=model,
        likelihood=likelihood,
        mll=mll,
        optimizer=optimizer,
        criterion_kwargs=criterion_kwargs,
    )


def get_poisson_model_trainer(
        dataloader,
        num_inducing,
        lr,
        learn_inducing_locations=True,
        model_kwargs=None,
        likelihood_kwargs=None,
        optimizer_kwargs=None,
):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    likelihood_kwargs = {} if likelihood_kwargs is None else likelihood_kwargs
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

    X = dataloader.dataset[:]["X"]
    if learn_inducing_locations:
        # initialize GP model
        inducing_points = (
            torch.rand((num_inducing, X.shape[-1]))
            * (X.max(dim=0).values - X.min(dim=0).values)
        ) + X.min(dim=0).values
    else:
        inducing_points = (
            (X - X.min(dim=0).values) / (X.max(dim=0).values - X.min(dim=0).values)
        )

    model = GPModel(
        inducing_points,
        learn_inducing_locations=learn_inducing_locations,
        **model_kwargs,
    )
    likelihood = PoissonLikelihood(**likelihood_kwargs)
    mll = VariationalELBO(likelihood, model, num_data=len(dataloader.dataset))
    criterion_kwargs = {}
    if likelihood.mean:
        criterion_kwargs["model_mean"] = "model_mean"

    # determine optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        **optimizer_kwargs,
    )
    model_trainer = GPModelTrainer(
        model=model,
        likelihood=likelihood,
        mll=mll,
        optimizer=optimizer,
        criterion_kwargs=criterion_kwargs,
    )
    return model_trainer


def read_poisson_gp_model_trainer(
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
    likelihood = PoissonLikelihood(log=log, mean=mean)
    likelihood.load_state_dict(checkpoint_info["likelihood"])
    mll = VariationalELBO(likelihood, model, num_data=len(dataloader.dataset))
    optimizer = optimizer(model.parameters())
    optimizer.load_state_dict(checkpoint_info["optimizer"])

    criterion_kwargs = {}
    if mean:
        criterion_kwargs["model_mean"] = "model_mean"

    return GPModelTrainer(
        model=model,
        likelihood=likelihood,
        mll=mll,
        optimizer=optimizer,
        criterion_kwargs=criterion_kwargs,
    )


def get_super_poisson_model_trainer(
        dataloader,
        num_inducing,
        lr,
        learn_inducing_locations=True,
        model_kwargs=None,
        likelihood_kwargs=None,
        optimizer_kwargs=None,
):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    likelihood_kwargs = {} if likelihood_kwargs is None else likelihood_kwargs
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

    X = dataloader.dataset[:]["X"]
    if learn_inducing_locations:
        # initialize GP model
        inducing_points = (
            torch.rand((num_inducing, X.shape[-1]))
            * (X.max(dim=0).values - X.min(dim=0).values)
        ) + X.min(dim=0).values
    else:
        inducing_points = (
            (X - X.min(dim=0).values) / (X.max(dim=0).values - X.min(dim=0).values)
        )

    model = GPModel(
        inducing_points,
        learn_inducing_locations=learn_inducing_locations,
        **model_kwargs,
    )
    likelihood = SuperPoissonLikelihood(**likelihood_kwargs)
    mll = VariationalELBO(likelihood, model, num_data=len(dataloader.dataset))
    criterion_kwargs = {"super_poisson_ratio": "super_poisson_ratio"}
    if likelihood.mean:
        criterion_kwargs["model_mean"] = "model_mean"

    # determine optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        **optimizer_kwargs,
    )
    model_trainer = GPModelTrainer(
        model=model,
        likelihood=likelihood,
        mll=mll,
        optimizer=optimizer,
        criterion_kwargs=criterion_kwargs,
    )
    return model_trainer


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

    criterion_kwargs = {"super_poisson_ratio": "super_poisson_ratio"}
    if mean:
        criterion_kwargs["model_mean"] = "model_mean"

    return GPModelTrainer(
        model=model,
        likelihood=likelihood,
        mll=mll,
        optimizer=optimizer,
        criterion_kwargs=criterion_kwargs,
    )
