import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        loss = None
        N = y.shape[0]
        M = x_scores - torch.gather(x_scores, dim=1, index=y.reshape(-1, 1)) + torch.tensor(self.delta)
        M_hinge = torch.max(M, torch.zeros(M.shape))
        L_i = M_hinge.sum(dim=1) - torch.gather(M_hinge, dim=1, index=y.reshape(-1, 1)).flatten()
        loss = L_i.sum(dim=0) / N
		
        self.grad_ctx['M'] = M
        self.grad_ctx['X'] = x
        self.grad_ctx['y'] = y

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """

        grad = None
        M = self.grad_ctx['M']
        X = self.grad_ctx['X']
        y = self.grad_ctx['y']
        N = y.shape[0]

        W_j = torch.as_tensor(M > 0, dtype=torch.int64)  # case: j != y
        W_y = W_j.sum(dim=1, keepdim=True)  # case: j = y, but with extra element and negated
        W_y = W_j - W_y  # case: j = y , discard extra element and remove negation.
        W_y = torch.gather(W_y, dim=1, index=y.reshape(-1, 1)).flatten()  # get vector of values from a matrix
        rows = torch.tensor(range(y.shape[0]))  # vector of rows
        indices = torch.LongTensor(list(zip(rows, y)))  # create indices for case j == y.
        G = W_j.index_put_(tuple(indices.t()), W_y)/N  # insert the special case from above to the matrix of j!=y
        grad = torch.matmul(X.t(), G)

        return grad
