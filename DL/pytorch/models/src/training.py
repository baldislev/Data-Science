import os
import abc
import sys
import torch
import torch.nn as nn
import tqdm.auto
from torch import Tensor
from typing import Any, Tuple, Callable, Optional, cast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.train_results import FitResult, BatchResult, EpochResult
from .classifier import Classifier

class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(
        self, model: nn.Module, device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs: int,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every: int = 1,
        post_epoch_fn: Callable = None,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """

        actual_num_epochs = 0
        epochs_without_improvement = 0

        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        best_acc = None

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.

            if print_every > 0 and (
                epoch % print_every == 0 or epoch == num_epochs - 1
            ):
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            train_result = self.train_epoch(dl_train=dl_train, verbose=verbose, **kw)
            train_loss += train_result.losses
            train_acc += [train_result.accuracy]

            test_result = self.test_epoch(dl_test=dl_test, verbose=verbose, **kw)
            test_loss += test_result.losses
            test_acc += [test_result.accuracy]
            actual_num_epochs += 1

            if best_acc is None or test_result.accuracy > best_acc:
                best_acc = test_result.accuracy
                epochs_without_improvement = 0
                if checkpoints is not None:
                    checkpoint_filename = f"{checkpoints}.pt"
                    self.save_checkpoint(checkpoint_filename=checkpoint_filename)
            else:
                epochs_without_improvement += 1
                if early_stopping is not None and epochs_without_improvement >= early_stopping:
                    break

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def save_checkpoint(self, checkpoint_filename: str):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        dirname = os.path.dirname(checkpoint_filename) or "."
        os.makedirs(dirname, exist_ok=True)
        torch.save({"model_state": self.model.state_dict()}, checkpoint_filename)
        print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and updates weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        if not verbose:
            pbar_file.close()

        return EpochResult(losses=losses, accuracy=accuracy)


class RNNTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.hidden_state = None

    def train_epoch(self, dl_train: DataLoader, **kw):
        self.hidden_state = None
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        self.hidden_state = None
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]
        h = self.hidden_state.detach() if self.hidden_state is not None else None
        self.optimizer.zero_grad()
        layer_output, self.hidden_state = self.model(x, hidden_state=h)

        loss = self.loss_fn(layer_output.transpose(1, 2), y)

        loss.backward()
        self.optimizer.step()

        y_pred = layer_output.argmax(dim=-1)
        num_correct = torch.sum(y_pred.eq(y))
        return BatchResult(loss.item(), num_correct.item() / seq_len)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        with torch.no_grad():
            h = self.hidden_state
            layer_output, self.hidden_state = self.model(x, hidden_state=h)

            loss = self.loss_fn(layer_output.transpose(1, 2), y)

            y_pred = layer_output.argmax(dim=-1)
            num_correct = torch.sum(y_pred.eq(y))

        return BatchResult(loss.item(), num_correct.item() / seq_len)


class VAETrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            optimizer: Optimizer,
            device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        :param model: Instance of the classifier model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        super().__init__(model, device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)
        self.optimizer.zero_grad()
        # forward:
        x_rec, z_mu, z_log_sigma2 = self.model(x)

        # loss
        loss, data_loss, _ = self.loss_fn(x, x_rec, z_mu, z_log_sigma2)

        #backward:
        loss.backward()

        #step:
        self.optimizer.step()

        return BatchResult(loss.item(), 1 / data_loss.item())

    def test_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)

        with torch.no_grad():
            x_rec, z_mu, z_log_sigma2 = self.model(x)
            loss, data_loss, _ = self.loss_fn(x, x_rec, z_mu, z_log_sigma2)

        return BatchResult(loss.item(), 1 / data_loss.item())

class ClassifierTrainer(Trainer):
    """
    Trainer for our Classifier-based models.
    """

    def __init__(
        self,
        model: Classifier,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        :param model: Instance of the classifier model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        super().__init__(model, device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        self.model: Classifier
        batch_loss: float
        num_correct: int

        y_score = self.model.forward(X)
        loss = self.loss_fn(y_score, y)
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        # step
        self.optimizer.step()
        # predict
        y_pred = self.model.classify_scores(y_score)
        num_correct = torch.sum(y == y_pred, dim=0).item()
        batch_loss = loss.item()
        return BatchResult(batch_loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        self.model: Classifier
        batch_loss: float
        num_correct: int

        with torch.no_grad():
            # forward
            y_score = self.model.forward(X)
            batch_loss = self.loss_fn(y_score, y).item()
            # predict
            y_pred = self.model.classify_scores(y_score)
            num_correct = torch.sum(y == y_pred, dim=0).item()
        return BatchResult(batch_loss, num_correct)


class LayerTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer):
        super().__init__(model)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_batch(self, batch) -> BatchResult:
        X, y = batch

        # forward:
        X_scores = self.model(X)
        loss = self.loss_fn(X_scores, y).item()

        # backward:
        self.optimizer.zero_grad()
        dloss = self.loss_fn.backward()
        self.model.backward(dloss)

        # optimization:
        self.optimizer.step()

        # accuracy evaluation:
        _, y_pred = torch.max(X_scores, dim=1)
        num_correct = torch.sum(y == y_pred, dim=0).item()

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch

        with torch.no_grad():
            X_scores = self.model(X)
            loss = self.loss_fn(X_scores, y).item()
            _, y_pred = torch.max(X_scores, dim=1)
            num_correct = torch.sum(y == y_pred, dim=0).item()
        return BatchResult(loss, num_correct)
