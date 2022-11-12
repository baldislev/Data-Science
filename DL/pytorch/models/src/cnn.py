import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS


POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.
    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()
        self.fcl = self._make_fcl()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        cur_in_channels = in_channels
        P = self.pool_every
        N = self.channels.__len__()
        idx = 0
        for _ in range(int(N/P)):
            for _ in range(P):
                layers += [
                    nn.Conv2d(in_channels=cur_in_channels,
                              out_channels=self.channels[idx],
                              **self.conv_params),
                    ACTIVATIONS[self.activation_type](**self.activation_params)
                ]
                cur_in_channels = self.channels[idx]
                idx += 1
            layers += [POOLINGS[self.pooling_type](**self.pooling_params)]
        for _ in range(N % P):
            layers += [
                nn.Conv2d(in_channels=cur_in_channels,
                          out_channels=self.channels[idx],
                          **self.conv_params),
                ACTIVATIONS[self.activation_type](**self.activation_params)
            ]
            cur_in_channels = self.channels[idx]
            idx += 1

        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            dummy_input = torch.zeros((1, *self.in_size))
            features_shape = self.feature_extractor(dummy_input).shape
            return features_shape[1] * features_shape[2] * features_shape[3]
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        mlp: MLP = None
        in_features = self._n_features()
        dims = self.hidden_dims
        nonlins = [ACTIVATIONS[self.activation_type](**self.activation_params)]*len(dims)
        mlp = MLP(
            in_dim=in_features,
            dims=dims,
            nonlins=nonlins
        )
        return mlp

    def _make_fcl(self):
        in_dim = self.hidden_dims[-1]
        out_dim = self.out_classes
        FCL = nn.Linear(in_dim, out_dim)
        return FCL

    def forward(self, x: Tensor):
        out: Tensor = None
        mlp_in = self.feature_extractor(x).reshape(x.shape[0], -1)
        mlp_out = self.mlp(mlp_in)
        out = self.fcl(mlp_out)
        return out


def get_conv_layers(in_channel: int, out_channel: int, kernel_size: int, padding: int, dropout, batchnorm, activation_type, activation_params):
    layers = [
        nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
    ]
    if dropout > 0:
        layers += [nn.Dropout2d(p=dropout)]
    if batchnorm:
        layers += [nn.BatchNorm2d(num_features=out_channel)]
    layers += [ACTIVATIONS[activation_type](**activation_params)]
    return layers


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        layers = []
        in_channel = in_channels
        for i in range(len(channels)-1):
            out_channel = channels[i]
            kernel_size = kernel_sizes[i]
            padding = int((kernel_size - 1) / 2)

            layers += get_conv_layers(in_channel, out_channel, kernel_size, padding, dropout, batchnorm,
                                      activation_type, activation_params)

            in_channel = out_channel

        padding = int((kernel_sizes[-1] - 1) / 2)
        layers += [
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=channels[-1],
                kernel_size=kernel_sizes[-1],
                padding=padding,
                bias=True
            )
        ]
        self.main_path = nn.Sequential(*layers)
        layer = []
        if in_channels != channels[-1]:
            layer += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channels[-1],
                    kernel_size=1,
                    bias=False
                )
            ]
        else:
            layer += [nn.Identity()]
        self.shortcut_path = nn.Sequential(*layer)
        # ========================

    def forward(self, x: Tensor):
        out: Tensor = None
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions, excluding the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->10.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        channels = [inner_channels[0], *inner_channels, in_out_channels]
        kernel_sizes = [1, *inner_kernel_sizes, 1]

        super().__init__(
            in_channels=in_out_channels,
            channels=channels,
            kernel_sizes=kernel_sizes,
            **kwargs
        )


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        cur_in_channels = in_channels
        i = 0
        P = self.pool_every
        while i < len(self.channels):
            num_conv_layers = min(P, len(self.channels) - i)
            is_residual = False
            is_bottleneck = self.bottleneck and cur_in_channels == self.channels[i:i + num_conv_layers][-1]
            if is_bottleneck:
                if P == num_conv_layers:
                    cur_channels = self.channels[i:i + num_conv_layers][1:-1]
                    cur_channels_len = len(cur_channels)
                    layers += [
                        ResidualBottleneckBlock(
                            in_out_channels=cur_in_channels,
                            inner_channels=cur_channels,
                            inner_kernel_sizes=[3] * cur_channels_len,
                            batchnorm=self.batchnorm,
                            dropout=self.dropout,
                            activation_type=self.activation_type,
                            activation_params=self.activation_params
                        )
                    ]
                else:
                    is_residual = True
            else:
                is_residual = True
            if is_residual:
                layers += [
                    ResidualBlock(
                        in_channels=cur_in_channels,
                        channels=self.channels[i:i + num_conv_layers],
                        kernel_sizes=[3] * num_conv_layers,
                        batchnorm=self.batchnorm,
                        dropout=self.dropout,
                        activation_type=self.activation_type,
                        activation_params=self.activation_params
                    )
                ]

            if num_conv_layers == P:
                layers += [POOLINGS[self.pooling_type](**self.pooling_params)]
            i += num_conv_layers
            cur_in_channels = self.channels[i-1]
        seq = nn.Sequential(*layers)
        return seq


class YourCNN(CNN):
    def __init__(self, *args, **kwargs):
        """
        See CNN.__init__
        """
        if 'conv_params' not in kwargs:
            kwargs['conv_params'] = dict(kernel_size=3, stride=1, padding=1)
        if 'activation_type' not in kwargs:
            kwargs['activation_type'] = 'lrelu'
        if 'activation_params' not in kwargs:
            kwargs['activation_params'] = dict(negative_slope=0.01)
        if 'pooling_type' not in kwargs:
            kwargs['pooling_type'] = 'avg'
        if 'pooling_params' not in kwargs:
            kwargs['pooling_params'] = dict(kernel_size=2)
        print(f'kwargs2: {kwargs}')
        super().__init__(*args, **kwargs)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        cur_in_channels = in_channels
        i = 0
        P = self.pool_every
        start = i
        skip_size = 2
        end = skip_size
        while i < len(self.channels):
            num_conv_layers = min(P, len(self.channels) - i)
            cur_channels = self.channels[i:i + num_conv_layers]
            cur_channels_len = len(cur_channels)
            while cur_channels_len > 1:
                layers += [
                    ResidualBlock(
                        in_channels=cur_in_channels,
                        channels=self.channels[start:end],
                        kernel_sizes=[3] * skip_size,
                        batchnorm=True,
                        dropout=0.15,
                        activation_type=self.activation_type,
                        activation_params=self.activation_params
                    )
                ]
                cur_channels_len -= skip_size
                cur_in_channels = self.channels[end - 1]
                start += skip_size
                end += skip_size

            if cur_channels_len == 1:
                layers += [
                    nn.Conv2d(
                        in_channels=cur_in_channels,
                        out_channels=self.channels[end - skip_size],
                        kernel_size=3,
                        bias=False
                    )
                ]

            if num_conv_layers == P:
                layers += [POOLINGS[self.pooling_type](**self.pooling_params)]
            i += num_conv_layers
            cur_in_channels = self.channels[i - 1]
        seq = nn.Sequential(*layers)
        return seq
