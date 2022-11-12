import abc
import torch


class Layer(abc.ABC):
    """
    A Layer is some computation element in a network architecture which
    supports automatic differentiation using forward and backward functions.
    """

    def __init__(self):
        # Store intermediate values needed to compute gradients in this hash
        self.grad_cache = {}
        self.training_mode = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes the forward pass of the layer.
        :param args: The computation arguments (implementation specific).
        :return: The result of the computation.
        """
        pass

    @abc.abstractmethod
    def backward(self, dout):
        """
        Computes the backward pass of the layer, i.e. the gradient
        calculation of the final network output with respect to each of the
        parameters of the forward function.
        :param dout: The gradient of the network with respect to the
        output of this layer.
        :return: A tuple with the same number of elements as the parameters of
        the forward function. Each element will be the gradient of the
        network output with respect to that parameter.
        """
        pass

    @abc.abstractmethod
    def params(self):
        """
        :return: Layer's trainable parameters and their gradients as a list
        of tuples, each tuple containing a tensor and it's corresponding
        gradient tensor.
        """
        pass

    def train(self, training_mode=True):
        """
        Changes the mode of this layer between training and evaluation (test)
        mode. Some layers have different behaviour depending on mode.
        :param training_mode: True: set the model in training mode. False: set
        evaluation mode.
        """
        self.training_mode = training_mode

    def __repr__(self):
        return self.__class__.__name__


class LeakyReLU(Layer):
    """
    Leaky version of Rectified linear unit.
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        if not (0 <= alpha < 1):
            raise ValueError("Invalid value of alpha")
        self.alpha = alpha

    def forward(self, x, **kw):
        """
        Computes max(alpha*x, x) for some 0<= alpha < 1.
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: ReLU of each sample in x.
        """

        out = torch.where(x > 0, x, self.alpha * x)
        self.grad_cache["x"] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, *).
        :return: Gradient with respect to layer input, shape (N, *)
        """
        x = self.grad_cache["x"]

        dz_dx = torch.where(x > 0.0, 1.0, self.alpha)
        dx = dout * dz_dx
        return dx

    def params(self):
        return []

    def __repr__(self):
        return f"LeakyReLU({self.alpha=})"


class ReLU(LeakyReLU):
    """
    Rectified linear unit.
    """

    def __init__(self):
        super().__init__(0.0)

    def __repr__(self):
        return "ReLU"


class Sigmoid(Layer):
    """
    Sigmoid activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes s(x) = 1/(1+exp(-x))
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: Sigmoid of each sample in x.
        """
        self.grad_cache['x'] = x
        out = 1 / (1 + torch.exp(-x))
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, *).
        :return: Gradient with respect to layer input, shape (N, *)
        """

        x = self.grad_cache['x']
        numenator = torch.exp(-x)
        denominator = torch.pow(1 + torch.exp(-x), 2)
        dz_dx = numenator / denominator
        dx = dout * dz_dx
        return dx

    def params(self):
        return []


class TanH(Layer):
    """
    Hyperbolic tangent activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: Sigmoid of each sample in x.
        """

        self.grad_cache['x'] = x
        out = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, *).
        :return: Gradient with respect to layer input, shape (N, *)
        """

        x = self.grad_cache['x']
        dz_dx = torch.pow(2 / (torch.exp(x) + torch.exp(-x)), 2)
        dx = dout * dz_dx
        return dx

    def params(self):
        return []


class Linear(Layer):
    """
    Fully-connected linear layer.
    """

    def __init__(self, in_features, out_features, wstd=0.1):
        """
        :param in_features: Number of input features (Din)
        :param out_features: Number of output features (Dout)
        :param wstd: standard deviation of the initial weights matrix
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        mean = torch.zeros((out_features, in_features))
        std = torch.full((out_features, in_features), wstd)
        self.w = torch.normal(mean, std)
        self.b = torch.zeros((1, out_features))
        # These will store the gradients
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def params(self):
        return [(self.w, self.dw), (self.b, self.db)]

    def forward(self, x, **kw):
        """
        Computes an affine transform, y = x W^T + b.
        :param x: Input tensor of shape (N,Din) where N is the batch
        dimension, and Din is the number of input features.
        :return: Affine transform of each sample in x.
        """

        x = x.reshape(x.shape[0], -1)
        out = (x @ self.w.t()) + self.b
        self.grad_cache["x"] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, Dout).
        :return: Gradient with respect to layer input, shape (N, Din)
        """
        x = self.grad_cache["x"]

        dz_dw = x
        dz_dx = self.w

        self.db += torch.sum(dout, axis=0)
        self.dw += dout.t() @ dz_dw
        dx = dout @ dz_dx
        return dx

    def __repr__(self):
        return f"Linear({self.in_features=}, {self.out_features=})"


class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Computes cross-entropy loss directly from class scores.
        Given class scores x, and a 1-hot encoding of the correct class yh,
        the cross entropy loss is defined as: -yh^T * log(softmax(x)).

        This implementation works directly with class scores (x) and labels
        (y), not softmax outputs or 1-hot encodings.

        :param x: Tensor of shape (N,D) where N is the batch
            dimension, and D is the number of features. Should contain class
            scores, NOT PROBABILITIES.
        :param y: Tensor of shape (N,) containing the ground truth label of
            each sample.
        :return: Cross entropy loss, as if we computed the softmax of the
            scores, encoded y as 1-hot and calculated cross-entropy by
            definition above. A scalar.
        """

        N = x.shape[0]

        # Shift input for numerical stability
        xmax, _ = torch.max(x, dim=1, keepdim=True)
        x = x - xmax

        real_scores = torch.gather(input=x, dim=1, index=y.reshape(-1, 1))
        loss_stacked = -real_scores + torch.log(torch.sum(torch.exp(x), dim=1, keepdim=True))
        loss = torch.sum(loss_stacked, dim=0) / N
        self.grad_cache["x"] = x
        self.grad_cache["y"] = y
        return torch.squeeze(loss)

    def backward(self, dout=1.0):
        """
        :param dout: Gradient with respect to layer output, a scalar which
            defaults to 1 since the output of forward is scalar.
        :return: Gradient with respect to layer input (only x), shape (N,D)
        """
        x = self.grad_cache["x"]
        y = self.grad_cache["y"]
        N = x.shape[0]

        dz_dx = torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)
        dz_dx[range(len(y)), y] += -1
        dx = dout * dz_dx
        dx = dx / N
        return dx

    def params(self):
        return []


class Dropout(Layer):
    def __init__(self, p=0.5):
        """
        Initializes a Dropout layer.
        :param p: Probability to drop an activation.
        """
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x, **kw):
        out = x
        if self.training_mode:
            num_features = out.shape[1]
            num_samples = out.shape[0]

            prob = self.p * torch.ones((num_features,))
            drop_val = torch.bernoulli(prob)
            drop_indices = (drop_val == 1).nonzero()
            # drop neurons:
            out[range(num_samples), drop_indices] = 0
            # save drop indices for backward pas:
            self.grad_cache["drop_indices"] = drop_indices
        else:  # test mode: scale activations
            out *= (1 - self.p)
        return out

    def backward(self, dout):
        if self.training_mode:
            num_samples = dout.shape[0]
            dx = dout
            drop_indices = self.grad_cache["drop_indices"]
            dx[range(num_samples), drop_indices] = 0
        else:
            dx = (1 - self.p) * dout
        return dx

    def params(self):
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential(Layer):
    """
    A Layer that passes input through a sequence of other layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x, **kw):
        out = x
        for layer in self.layers:
            out = layer.forward(out, **kw)
        return out

    def backward(self, dout):
        din = None

        for layer in reversed(self.layers):
            din = layer.backward(dout)
            dout = din
        return din

    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params()
        return params

    def train(self, training_mode=True):
        for layer in self.layers:
            layer.train(training_mode)

    def __repr__(self):
        res = "Sequential\n"
        for i, layer in enumerate(self.layers):
            res += f"\t[{i}] {layer}\n"
        return res

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, item):
        return self.layers[item]


class MLP(Layer):
    """
    A simple multilayer perceptron based on our custom Layers.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """

    def __init__(
            self,
            in_features,
            num_classes,
            hidden_features=(),
            activation="relu",
            dropout=0,
            **kw,
    ):
        super().__init__()
        """
        Create an MLP model Layer.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        layers = []

        layers.append(Linear(in_features, hidden_features[0]))
        layers.append(ReLU() if activation == 'relu' else Sigmoid())
        if dropout > 0:
            layers.append(Dropout(dropout))

        n = len(hidden_features)
        for i in range(n):
            if i == n - 1:
                layers.append(Linear(hidden_features[-1], num_classes))
            else:
                layers.append(Linear(hidden_features[i], hidden_features[i + 1]))
                layers.append(ReLU() if activation == 'relu' else Sigmoid())
                if dropout > 0:
                    layers.append(Dropout(dropout))
        self.sequence = Sequential(*layers)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f"MLP, {self.sequence}"
