import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import utils.dataloader_utils as dataloader_utils

from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train = None
        y_train = None
        for idx, (x, y) in enumerate(dl_train):
            if x_train is None:
                x_train = x
                y_train = y
            else:
                x_train = torch.cat((x_train, x), dim=0)
                y_train = torch.cat((y_train, y), dim=0)

        n_classes = len(torch.unique(y_train))

        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        for i in range(n_test):
            neighbors = dist_matrix[:, i].topk(self.k, largest=False)
            votes = self.y_train[neighbors.indices]
            y_pred[i] = torch.mode(votes).values

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    dists = None
    sq1 = torch.sum(torch.pow(x1, 2), 1, keepdim=True)
    sq2 = torch.sum(torch.pow(x2, 2), 1, keepdim=False)
    Cov = torch.matmul(x1, x2.transpose(0, 1))
    dists = sq1 - 2*Cov + sq2
    dists = torch.sqrt(dists)

    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    accuracy = None
    accuracy = (y == y_pred).sum().item()/y.size(0)

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []
    tot_len = len(ds_train)
    valid_len = int(tot_len/num_folds)
    indices = list(range(tot_len))
    np.random.shuffle(indices)
    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        acc = []
        for fold in range(num_folds):
            valid_indices = indices[fold*valid_len : (fold+1)*valid_len]  # list(range(fold*valid_len, (fold+1)*valid_len))
            train_indices = list(set(indices).difference(set(valid_indices)))
            train_sampler = dataloaders.IndicesSampler(train_indices)
            valid_sampler = dataloaders.IndicesSampler(valid_indices)

            dl_train = torch.utils.data.DataLoader(ds_train, sampler=train_sampler)
            dl_valid = torch.utils.data.DataLoader(ds_train, sampler=valid_sampler)
            x_valid, y_valid = data_from_loader(dl_valid)
            model.train(dl_train)
            y_pred = model.predict(x_valid)
            acc.append(accuracy(y_valid, y_pred))

        accuracies.append(acc)

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies

def data_from_loader(dl):
    x_data = None
    y_data = None
    for idx, (x, y) in enumerate(dl):
        if x_data is None:
            x_data = x
            y_data = y
        else:
            x_data = torch.cat((x_data, x), dim=0)
            y_data = torch.cat((y_data, y), dim=0)
    return x_data, y_data
