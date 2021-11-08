from abc import ABC, abstractmethod
import time

import dill
import gpytorch
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, Checkpoint, EarlyStopping, TerminateOnNan
from ignite.contrib.handlers import ProgressBar, global_step_from_engine
import torch
from threadpoolctl import threadpool_limits

from amulator.training.data import DictionaryDataset


# global metrics to use for Engine
def running_avg_loss(engine):
    return -engine.state.metrics["running_avg_loss"]


class ModelTrainer(ABC):
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    @abstractmethod
    def train_step(self, engine, batch):
        raise NotImplementedError


# https://discuss.pytorch.org/t/ignite-correct-way-of-using-the-library-how-to-pass-model-to-callable/74522/2
class GPModelTrainer(ModelTrainer):
    def __init__(
            self,
            model,
            likelihood,
            mll,
            optimizer,
            *args,
            **kwargs
    ):
        super().__init__(model=model, criterion=None, optimizer=optimizer)
        # include likelihood to load model from checkpoint
        self.likelihood = likelihood
        # GPModel objective is -mll
        self.mll = mll
        self.losses = []

    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        # our GP model uses Dictionarydataset to allow extra kwargs to mll
        X = batch["X"]
        y = batch["y"]
        criterion_kwargs = {k: batch[k] for k in batch.keys() - {"X", "y"}}

        y_pred = self.model(X)
        # criterion is given by marginal likelihood => make negative
        loss = -self.mll(y_pred, y, **criterion_kwargs)
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        return loss.item()


def get_trainer_engine(
        model_trainer,
        filename_prefix,
        save_dir,
        save_every=10,
        num_saved=10,
        require_empty=False,
        create_dir=True,
        patience=10,
        **tqdm_kwargs
):
    """Return trainer_engine based on model_trainer with NaN termination,
    loss logging, early stopping and progress bar.

    Parameters
    ----------
    model_trainer : GPModelTrainer
        keeps track of model, loss and optimizer
    filename_prefix : str
        prefix for saved checkpoint
    save_dir : str [Default: %Y%m%d of run start]
        directory to save checkpoints to
    save_every : int
        number of training intervals to save checkpoints after
    num_saved : int
        maximum number of checkpoints to keep
    require_empty : bool
        require save_dir to not contain '.pt' files
    create_dir : bool
        create save_dir if it does not exist
    patience : int
        number of events to wait if no improvement and then stop the training

    Returns
    -------
    trainer_engine : ignite.engine.Engine
        Engine for model_trainer
    """
    trainer_engine = Engine(model_trainer.train_step)

    # terminate training on NaN value to prevent unnecessary loops
    trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # calculate the running average of the loss function
    avg_output = RunningAverage(output_transform=lambda x: x)
    avg_output.attach(trainer_engine, "running_avg_loss")

    stop = EarlyStopping(patience=patience, score_function=running_avg_loss, trainer=trainer_engine)
    trainer_engine.add_event_handler(Events.COMPLETED, stop)

    # add progress bar
    pbar = ProgressBar(**tqdm_kwargs)
    pbar.attach(
        trainer_engine,
        ["running_avg_loss"],
        event_name=Events.EPOCH_COMPLETED,
        closing_event_name=Events.COMPLETED,
    )

    # add checkpoint saving
    to_save = {
        "trainer": trainer_engine,
        "model": model_trainer.model,
        "likelihood": model_trainer.likelihood,
        "optimizer": model_trainer.optimizer,
    }

    handler = ModelCheckpoint(
        save_dir,
        filename_prefix,
        n_saved=num_saved,
        create_dir=create_dir,
        require_empty=require_empty,
        global_step_transform=global_step_from_engine(trainer_engine, Events.EPOCH_COMPLETED),
        score_function=running_avg_loss,
        score_name="avg_epoch_loss",
        include_self=True,
    )
    trainer_engine.add_event_handler(
        Events.EPOCH_COMPLETED(every=save_every) | Events.COMPLETED,
        handler,
        to_save,
    )

    best_handler = Checkpoint(
        to_save,
        save_dir,
        n_saved=num_saved,
        filename_prefix=f"{filename_prefix}_best",
        score_name="avg_epoch_loss",
        global_step_transform=global_step_from_engine(trainer_engine)
    )
    trainer_engine.add_event_handler(Events.COMPLETED, best_handler)

    return trainer_engine


def train_model(
        # dataloader contains x, y, *criterion_vals
        dataloader,
        model_trainer,
        max_epochs=150,
        save_prefix=time.strftime("%H%M"),
        save_suffix=None,
        save_dir=time.strftime("%Y%m%d"),
        save_every=100,
        num_saved=10,
        patience=10,
        require_empty=False,
        create_dir=True,
        trainer_engine=None,
        num_threads=None,
):
    """Train model_trainer on given dataloader.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        loader for the data to be passed to model_trainer
    model_trainer : GPModelTrainer
        keeps track of model, loss and optimizer
    max_epochs : int
        maximum number of epochs to train
    save_prefix : str [Default: %H%M of run start]
        prefix for saved checkpoint
    save_suffix : Optional[str]
        optional suffix to append to prefix
    save_dir : str [Default: %Y%m%d of run start]
        directory to save checkpoints to
    save_every : int
        number of training intervals to save checkpoints after
    num_saved : int
        maximum number of checkpoints to keep
    patience : int
        number of events to wait if no improvement and then stop the training
    require_empty : bool
        require save_dir to not contain '.pt' files
    create_dir : bool
        create save_dir if it does not exist
    trainer_engine : Optional[ignite.engine.Engine]
        engine without ModelCheckpoint handler with running_avg_loss metric
    num_threads : Optional[int]
        limit number of threads with threadpoolctl

    Returns
    -------
    trainer_engine : ignite.engine.Engine
        trained engine

    save checkpoints to
    {save_dir}/{save_prefix}_{model_name}_{likelihood_name}_optim_{optimizer_name}_{save_suffix}
    """
    if save_suffix is None:
        save_suffix = ""
    else:
        save_suffix = f"_{save_suffix}"
    model_name = type(model_trainer.model).__name__
    likelihood_name = type(model_trainer.likelihood).__name__
    optimizer_name = type(model_trainer.optimizer).__name__
    model_info = f"{model_name}_{likelihood_name}_optim_{optimizer_name}"
    filename_prefix = f"{save_prefix}_{model_info}{save_suffix}"

    if trainer_engine is None:
        trainer_engine = get_trainer_engine(
            model_trainer=model_trainer,
            filename_prefix=filename_prefix,
            save_dir=save_dir,
            save_every=save_every,
            num_saved=num_saved,
            require_empty=require_empty,
            create_dir=create_dir,
            patience=patience,
        )

    with threadpool_limits(limits=num_threads):
        trainer_engine.run(dataloader, max_epochs=max_epochs)

    # save full model
    with open(f"{save_dir}/{filename_prefix}_full_model_{max_epochs}.pt", "wb") as f:
        dill.dump(
            {
                "dataloader": dataloader,
                "model_trainer": model_trainer,
                "trainer_engine": trainer_engine,
            },
            f
        )

    return trainer_engine
