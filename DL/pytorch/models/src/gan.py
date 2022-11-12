import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        modules = []
        out_features = 1
        in_channels = self.in_size[0]
        channels_num = [128, 256, 512, 1024]
        conv_params = dict(kernel_size=5, stride=2, padding=2, bias=False)
        for i, channels in enumerate(channels_num):
            if i == 0:
                modules += [
                    nn.Conv2d(in_channels=in_channels, out_channels=channels, **conv_params),
                    nn.LeakyReLU(negative_slope=0.2)
                ]
            else:
                modules += [
                    nn.Conv2d(in_channels=in_channels, out_channels=channels, **conv_params),
                    nn.BatchNorm2d(num_features=channels),
                    nn.LeakyReLU(negative_slope=0.2)
                ]
            in_channels = channels

        cnn = nn.Sequential(*modules)
        cnn_features = self.calc_num_cnn_features(cnn, in_size)
        modules += [nn.Flatten(start_dim=1),
                    nn.Linear(cnn_features, out_features)]
        self.disc = nn.Sequential(*modules)

    def calc_num_cnn_features(self, cnn, in_size):
        with torch.no_grad():
            x = torch.randn(1, *in_size)
            out_shape = cnn(x).shape[1:]
        return out_shape[0] * out_shape[1] * out_shape[2]

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        y = self.disc(x)
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        self.featuremap_size = featuremap_size
        in_channels = 1024
        first_layer_features = (in_channels*featuremap_size*featuremap_size)
        modules = []
        modules += [nn.Linear(z_dim, first_layer_features),
                    nn.Unflatten(1, (in_channels, featuremap_size, featuremap_size)),
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU()]

        channels_num = [512, 256, 128, out_channels]
        conv_params = dict(kernel_size=5, padding=2, stride=2, output_padding=1, bias=False)
        for i, channels in enumerate(channels_num):
            if i != len(channels_num) - 1:
                modules += [
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=channels, **conv_params),
                    nn.BatchNorm2d(num_features=channels),
                    nn.ReLU()
                ]
            else:
                modules += [
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=channels, **conv_params),
                    nn.Tanh()
                ]
            in_channels = channels

        self.gen = nn.Sequential(*modules)

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        sample = torch.randn(n, self.z_dim).to(device=device)
        if with_grad:
            samples = self.forward(sample)
        else:
            with torch.no_grad():
                samples = self.forward(sample)
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        x = self.gen(z)
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    loss_fn = nn.BCEWithLogitsLoss()

    data_l = data_label - label_noise/2
    data_r = data_label + label_noise/2

    generated_l = (1-data_label) - label_noise/2
    generated_r = (1-data_label) + label_noise/2

    noisy_data = torch.FloatTensor(y_data.shape[0]).uniform_(data_l, data_r).reshape(y_data.shape).to(device=y_data.device)
    noisy_generated = torch.FloatTensor(y_generated.shape[0]).uniform_(generated_l, generated_r).reshape(y_generated.shape).to(device=y_generated.device)

    loss_data = loss_fn(y_data, noisy_data)
    loss_generated = loss_fn(y_generated, noisy_generated)
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    loss_fn = torch.nn.BCEWithLogitsLoss()
    labels = torch.ones_like(y_generated) * data_label
    loss = loss_fn(y_generated, labels)
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """
    n = x_data.shape[0]

    dsc_model.train(True)
    gen_model.train(True)

    dsc_optimizer.zero_grad()
    sample = gen_model.sample(n, with_grad=False)
    dsc_fake = dsc_model(sample)
    dsc_real = dsc_model(x_data)
    dsc_loss = dsc_loss_fn(dsc_real, dsc_fake)
    dsc_loss.backward()
    dsc_optimizer.step()

    gen_optimizer.zero_grad()
    sample = gen_model.sample(n, with_grad=True)
    dsc_fake = dsc_model(sample)
    gen_loss = gen_loss_fn(dsc_fake)
    gen_loss.backward()
    gen_optimizer.step()

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    if len(gen_losses) % 5 == 0:
        torch.save(gen_model, checkpoint_file)
        saved = True

    return saved
