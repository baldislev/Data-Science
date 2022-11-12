import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from torch import Tensor, nn
from typing import Optional
from sklearn.metrics import roc_curve


class Classifier(nn.Module, ABC):
    """
    Wraps a model which produces raw class scores, and provides methods to compute
    class labels and probabilities.
    """

    def __init__(self, model: nn.Module):
        """
        :param model: The wrapped model. Should implement a `forward()` function
        returning (N,C) tensors where C is the number of classes.
        """
        super().__init__()
        self.model = model

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (N, D) input tensor, N samples with D features
        :returns: (N, C) i.e. C class scores for each of N samples
        """
        z: Tensor = None
        z = self.model(x)
        assert z.shape[0] == x.shape[0] and z.ndim == 2, "raw scores should be (N, C)"
        return z

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        :param x: (N, D) input tensor, N samples with D features
        :returns: (N, C) i.e. C probability values between 0 and 1 for each of N
            samples.
        """
        z = self.forward(x)
        return self.predict_proba_scores(z)

    def predict_proba_scores(self, z: Tensor) -> Tensor:
        """
        :param z: (N, C) scores tensor, e.g. calculated by this model.
        :returns: (N, C) i.e. C probability values between 0 and 1 for each of N
            samples.
        """
        return self.softmax(z)

    def classify(self, x: Tensor) -> Tensor:
        """
        :param x: (N, D) input tensor, N samples with D features
        :returns: (N,) tensor of type torch.int containing predicted class labels.
        """
        # Calculate the class probabilities
        y_proba = self.predict_proba(x)
        # Use implementation-specific helper to assign a class based on the
        # probabilities.
        return self._classify(y_proba)

    def classify_scores(self, z: Tensor) -> Tensor:
        """
        :param z: (N, C) scores tensor, e.g. calculated by this model.
        :returns: (N,) tensor of type torch.int containing predicted class labels.
        """
        y_proba = self.predict_proba_scores(z)
        return self._classify(y_proba)

    @abstractmethod
    def _classify(self, y_proba: Tensor) -> Tensor:
        pass


class ArgMaxClassifier(Classifier):
    """
    Multiclass classifier that chooses the maximal-probability class.
    """

    def __init__(
            self, model: nn.Module
    ):
        """
        :param model: The wrapped model. Should implement a `forward()` function
        """
        super().__init__(model)

    def _classify(self, y_proba: Tensor):
        out = torch.argmax(y_proba, dim=1).view(-1)
        return out

class BinaryClassifier(Classifier):
    """
    Binary classifier which classifies based on thresholding the probability of the
    positive class.
    """

    def __init__(
            self, model: nn.Module, positive_class: int = 1, threshold: float = 0.5
    ):
        """
        :param model: The wrapped model. Should implement a `forward()` function
        returning (N,C) tensors where C is the number of classes.
        :param positive_class: The index of the 'positive' class (the one that's
            thresholded to produce the class label '1').
        :param threshold: The classification threshold for the positive class.
        """
        super().__init__(model)
        assert positive_class in (0, 1)
        assert 0 < threshold < 1
        self.threshold = threshold
        self.positive_class = positive_class

    def _classify(self, y_proba: Tensor):
        y = (y_proba[:, self.positive_class] > self.threshold).to(torch.int32)
        return y


def plot_decision_boundary_2d(
        classifier: Classifier,
        x: Tensor,
        y: Tensor,
        dx: float = 0.1,
        ax: Optional[plt.Axes] = None,
        cmap=plt.cm.get_cmap("coolwarm"),
):
    """
    Plots a decision boundary of a classifier based on two input features.
    :param classifier: The classifier to use.
    :param x: The (N, 2) feature tensor.
    :param y: The (N,) labels tensor.
    :param dx: Step size for creating an evaluation grid.
    :param ax: Optional Axes to plot on. If None, a new figure with one Axes will be
        created.
    :param cmap: Colormap to use.
    :return: A (figure, axes) tuple.
    """
    assert x.ndim == 2 and y.ndim == 1
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig, ax = ax.get_figure(), ax

    # Plot the data
    ax.scatter(
        x[:, 0].numpy(),
        x[:, 1].numpy(),
        c=y.numpy(),
        s=20,
        alpha=0.8,
        edgecolor="k",
        cmap=cmap,
    )

    
    x1_grid, x2_grid, y_hat = None, None, None
    x_min, x_max = x[:, 0].min() - 10 * dx, x[:, 0].max() + 10 * dx
    y_min, y_max = x[:, 1].min() - 10 * dx, x[:, 1].max() + 10 * dx
    xs = torch.arange(x_min, x_max, dx)
    ys = torch.arange(y_min, y_max, dx)
    x1_grid, x2_grid = torch.meshgrid(xs, ys)
	
    a = torch.stack([x1_grid, x2_grid]).reshape(2, -1)
    y_hat = classifier.classify(a.T).reshape(x1_grid.shape)

    # Plot the decision boundary as a filled contour
    ax.contourf(x1_grid.numpy(), x2_grid.numpy(), y_hat.numpy(), alpha=0.3, cmap=cmap)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    return fig, ax


def select_roc_thresh(
        classifier: Classifier, x: Tensor, y: Tensor, plot: bool = False):
    """
    Calculates (and optionally plot) a classification threshold of a binary
    classifier, based on ROC analysis.
    :param classifier: The BINARY classifier to use.
    :param x: The (N, D) feature tensor.
    :param y: The (N,) labels tensor.
    :param plot: Whether to also create the ROC plot.
    :param ax: If plotting, the ax to plot on. If not provided a new figure will be
        created.
    """

    y_proba = classifier.predict_proba(x)[:, classifier.positive_class]
    fpr, tpr, thresh = roc_curve(y_true=y.detach().numpy(), y_score=y_proba.detach().numpy())
    n = len(thresh)
    optimal_thresh_idx = torch.argmin(torch.tensor([(1 - tpr[i]) ** 2 + (fpr[i]) ** 2 for i in range(n)])).item()
    optimal_thresh = thresh[optimal_thresh_idx]

    # scikit will add +1 to the highest threshold value, it's a convention, so one should discard the addition.
    optimal_thresh = optimal_thresh if optimal_thresh <= 1 else optimal_thresh - 1

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(fpr, tpr, color="C0")
        ax.scatter(
            fpr[optimal_thresh_idx], tpr[optimal_thresh_idx], color="C1", marker="o"
        )
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR=1-FNR")
        ax.legend(["ROC", f"Threshold={optimal_thresh:.2f}"])

    return optimal_thresh
