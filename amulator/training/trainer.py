from abc import ABC, abstractmethod
import time

import dill
import gpytorch
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNan,
)
from ignite.contrib.handlers import LRScheduler, ProgressBar, global_step_from_engine
import torch
from torch.optim.lr_scheduler import (
    ExponentialLR,
)
from threadpoolctl import threadpool_limits

from amulator.training.data import DictionaryDataset


# global metrics to use for Engine
def running_avg_loss(engine):
    return -engine.state.metrics["running_avg_loss"]


def running_avg_mll(engine):
    return engine.state.metrics["running_avg_mll"]


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
        self.eval_losses = []

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

    def eval_mll(self, engine, batch):
        self.model.eval()

        # our GP model uses Dictionarydataset to allow extra kwargs to mll
        X = batch["X"]
        y = batch["y"]
        criterion_kwargs = {k: batch[k] for k in batch.keys() - {"X", "y"}}

        with torch.no_grad():
            y_pred = self.model(X)
            mll = self.mll(y_pred, y, **criterion_kwargs)
            self.eval_losses.append(mll.item())

        return mll.item()


def get_trainer_engine(
        model_trainer,
        filename_prefix,
        save_dir,
        save_every=10,
        num_saved=10,
        require_empty=False,
        create_dir=True,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        save_history=True,
        tqdm_kwargs=None,
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

    # add progress bar
    tqdm_kwargs = {} if tqdm_kwargs is None else tqdm_kwargs
    pbar = ProgressBar(**tqdm_kwargs)
    pbar.attach(
        trainer_engine,
        ["running_avg_loss"],
        event_name=Events.EPOCH_COMPLETED,
        closing_event_name=Events.COMPLETED,
    )

    # add checkpoint saving
    to_save = {
        # # trainer saves some state that cannot be loaded with torch.load?
        # "trainer": trainer_engine,
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

    if lr_scheduler is not None:
        lr_scheduler_kwargs = {} if lr_scheduler_kwargs is None else lr_scheduler_kwargs
        scheduler = lr_scheduler(optimizer=model_trainer.optimizer, **lr_scheduler_kwargs)
        lr_handler = LRScheduler(scheduler, save_history=save_history)
        trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, lr_handler)

    return trainer_engine


def get_evaluator_engine(
        model_trainer,
        trainer_engine,
        filename_prefix,
        save_dir,
        num_saved=10,
        require_empty=False,
        create_dir=True,
        patience=10,
):
    """Return evaluator_engine based on model_trainer.

    Parameters
    ----------
    model_trainer : GPModelTrainer
        keeps track of model, loss and optimizer
    trainer_engine : ignite.engine.Engine
        Engine for model_trainer
    filename_prefix : str
        prefix for saved checkpoint
    save_dir : str [Default: %Y%m%d of run start]
        directory to save checkpoints to
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
    evaluator_engine : ignite.engine.Engine
        Engine for model_trainer performance evaluation
    """
    evaluator_engine = Engine(model_trainer.eval_mll)
    # src=None: use output of evaluator engine as input to running_average
    # output_transform:
    loss_metric = RunningAverage(
        src=None,
        output_transform=lambda output: output,
    )
    loss_metric.attach(evaluator_engine, "running_avg_mll")

    to_save = {
        "model": model_trainer.model,
    }

    best_handler = Checkpoint(
        to_save,
        DiskSaver(
            save_dir,
            create_dir=create_dir,
            require_empty=require_empty,
        ),
        n_saved=num_saved,
        filename_prefix=f"{filename_prefix}_best",
        score_name="running_avg_mll",
        global_step_transform=global_step_from_engine(evaluator_engine),
    )
    evaluator_engine.add_event_handler(Events.COMPLETED, best_handler)

    if patience is not None:
        # add early stopping based on evaluator performance
        stop = EarlyStopping(patience=patience, score_function=running_avg_mll, trainer=trainer_engine)
        evaluator_engine.add_event_handler(Events.COMPLETED, stop)

    return evaluator_engine


def train_model(
        # dataloader contains x, y, *criterion_vals
        train_loader,
        model_trainer,
        max_epochs=150,
        save_dir=time.strftime("%Y_%m_%d_%H_%M_%S"),
        save_prefix=None,
        save_suffix=None,
        filename_prefix=None,
        save_every=100,
        num_saved=10,
        require_empty=False,
        create_dir=True,
        eval_loader=None,
        patience=None,
        trainer_engine=None,
        num_threads=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        save_history=True,
):
    """Train model_trainer on given dataloader.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        loader for the training data to be passed to model_trainer
    model_trainer : GPModelTrainer
        keeps track of model, loss and optimizer
    max_epochs : int
        maximum number of epochs to train
    save_dir : str [Default: %Y_%m_%d_%H_%M_%S of run start]
        directory to save checkpoints to
    save_prefix : Optional[str] [Default: None]
        prefix for saved checkpoint
    save_suffix : Optional[str]
        optional suffix to append to prefix
    filename_prefix : Optional[str]
        if given, determines full filename prefix, only appends checkpoint info
    save_every : int
        number of training intervals to save checkpoints after
    num_saved : int
        maximum number of checkpoints to keep
    require_empty : bool
        require save_dir to not contain '.pt' files
    create_dir : bool
        create save_dir if it does not exist
    eval_loader : Optional[torch.utils.data.DataLoader]
        loader for the evaluation data to be passed to model_trainer
    patience : int
        number of events to wait if no improvement in evaluation and then stop the training
    trainer_engine : Optional[ignite.engine.Engine]
        engine without ModelCheckpoint handler with running_avg_loss metric
    num_threads : Optional[int]
        limit number of threads with threadpoolctl
    lr_scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        schedule for learning rate updates
    lr_scheduler_kwargs : dict
        kwargs for lr_scheduler
    save_history : Optional[bool]
        save lr param history

    Returns
    -------
    trainer_engine : ignite.engine.Engine
        trained engine

    save checkpoints to
    {save_dir}/{save_prefix}_{model_name}_{likelihood_name}_optim_{optimizer_name}_{save_suffix}
    """
    if filename_prefix is None:
        if save_suffix is None:
            save_suffix = ""
        else:
            save_suffix = f"_{save_suffix}"
        if save_prefix is None:
            save_prefix = ""
        else:
            save_prefix = f"{save_prefix}_"

        model_name = type(model_trainer.model).__name__
        likelihood_name = type(model_trainer.likelihood).__name__
        optimizer_name = type(model_trainer.optimizer).__name__
        model_info = f"{model_name}_{likelihood_name}_optim_{optimizer_name}"
        filename_prefix = f"{save_prefix}{model_info}{save_suffix}"

    if trainer_engine is None:
        trainer_engine = get_trainer_engine(
            model_trainer=model_trainer,
            filename_prefix=filename_prefix,
            save_dir=save_dir,
            save_every=save_every,
            num_saved=num_saved,
            require_empty=require_empty,
            create_dir=create_dir,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            save_history=save_history,
        )

    if eval_loader is not None:
        eval_engine = get_evaluator_engine(
            model_trainer=model_trainer,
            trainer_engine=trainer_engine,
            filename_prefix=filename_prefix,
            save_dir=save_dir,
            num_saved=num_saved,
            require_empty=require_empty,
            create_dir=create_dir,
            patience=patience,
        )
        # add evaluator to trainer
        @trainer_engine.on(Events.EPOCH_COMPLETED)
        def run_validation():
            eval_engine.run(eval_loader)

    with threadpool_limits(limits=num_threads):
        trainer_engine.run(train_loader, max_epochs=max_epochs)

    # save full model
    with open(f"{save_dir}/{filename_prefix}_full_model_{max_epochs}.pt", "wb") as f:
        dill.dump(
            {
                "dataloader": train_loader,
                "model_trainer": model_trainer,
                "trainer_engine": trainer_engine,
            },
            f
        )

    if eval_loader is None:
        return trainer_engine
    else:
        return trainer_engine, eval_engine
