import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        self.weights = None
        mean = torch.zeros((n_features, n_classes))
        std = torch.full((n_features, n_classes), weight_std)
        self.weights = torch.normal(mean=mean, std=std)

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        y_pred, class_scores = None, None
        class_scores = torch.matmul(x, self.weights)
        y_pred = torch.argmax(class_scores, dim=-1)

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """
        acc = None
        acc = (y == y_pred).sum().item() / y.shape[0]

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            # Train loop:
            loss = 0
            acc = 0
            for batch, labels in dl_train:
                pred_labels, x_scores = self.predict(batch)
                reg_loss = (weight_decay/2)*torch.pow(torch.linalg.norm(self.weights, ord='fro'), 2)
                loss += loss_fn.loss(batch, labels, x_scores, pred_labels) + reg_loss
                acc += self.evaluate_accuracy(labels, pred_labels)
                self.weights = self.weights - learn_rate*(loss_fn.grad() + weight_decay*self.weights)

            average_loss = loss.item()/len(dl_train)
            total_correct = acc/len(dl_train)
            train_res.loss.append(average_loss)
            train_res.accuracy.append(total_correct)

            # Validation loop:
            loss = 0
            acc = 0
            for batch, labels in dl_valid:
                pred_labels, x_scores = self.predict(batch)
                reg_loss = (weight_decay/2)*torch.pow(torch.linalg.norm(self.weights, ord='fro'), 2)
                x_scores = torch.matmul(batch, self.weights)
                loss += loss_fn.loss(batch, labels, x_scores, pred_labels) + reg_loss
                acc += self.evaluate_accuracy(labels, pred_labels)

            average_loss = loss.item()/len(dl_valid)
            total_correct = acc/len(dl_valid)
            valid_res.loss.append(average_loss)
            valid_res.accuracy.append(total_correct)
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """
        C, H, W = img_shape
        w = self.weights[1:, :] if has_bias else self.weights
        w_images = w.t().reshape((self.n_classes, C, H, W))
		
        return w_images


def hyperparams():
    hp = dict(weight_std=0.01, learn_rate=0.1, weight_decay=0.001)

    return hp
