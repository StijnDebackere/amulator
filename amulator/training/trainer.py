from abc import ABC, abstractmethod
import time

import gpytorch
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.contrib.handlers import ProgressBar, global_step_from_engine
import torch
from threadpoolctl import threadpool_limits

from amulator.training.data import DictionaryDataset


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



        return y_pred, y

    @classmethod
    def load(cls, fname, model_cls, likelihood_cls, optimizer_cls, mll_cls):
        trainer_info = torch.load(fname)
        model = model[""]
        return cls(
            model=model,
            likelihood=likelihood,
            mll=mll,
            optimizer=optimizer,
        )



def train_model(
        # dataloader contains x, y, *criterion_vals
        dataloader,
        model_trainer,
        save_prefix=time.strftime("%H%M"),
        save_dir=time.strftime("%Y%m%d"),
        num_threads=None,
        max_epochs=150,
        save_every=100,
        n_saved=10,
        require_empty=True,
        create_dir=True,
        patience=10,
        log_filename=None,
):
    trainer_engine = Engine(model_trainer.train_step)

    # terminate training on NaN value to prevent long loops
    trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # calculate the running average of the loss function
    avg_output = RunningAverage(output_transform=lambda x: x)
    avg_output.attach(trainer_engine, "running_avg_loss")

    def score_function(engine):
        return -engine.state.metrics["running_avg_loss"]

    stop = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer_engine)
    trainer_engine.add_event_handler(Events.COMPLETED, stop)

    # add checkpoint saving
    handler = ModelCheckpoint(
        save_dir,
        save_prefix,
        n_saved=n_saved,
        create_dir=create_dir,
        require_empty=require_empty,
        global_step_transform=global_step_from_engine(trainer_engine, Events.EPOCH_COMPLETED),
        score_function=score_function,
        score_name="avg_epoch_loss",
        include_self=True,
    )
    trainer_engine.add_event_handler(
        Events.EPOCH_COMPLETED(every=save_every),
        handler,
        {
            "model": model_trainer.model,
            "likelihood": model_trainer.likelihood,
            "mll": model_trainer.mll,
            "optimizer": model_trainer.optimizer,
        },
    )

    # add progress bar
    tqdm_kwargs = {"persist": False}
    if log_filename is not None:
        log_file = open(log_filename, "w")
        tqdm_kwargs["file"] = log_file

    pbar = ProgressBar(**tqdm_kwargs)
    pbar.attach(
        trainer_engine,
        ["running_avg_loss"],
        event_name=Events.EPOCH_COMPLETED,
        closing_event_name=Events.COMPLETED,
    )

    with threadpool_limits(limits=num_threads):
        trainer_engine.run(dataloader, max_epochs=max_epochs)

    return trainer_engine
