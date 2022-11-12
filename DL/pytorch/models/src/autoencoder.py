import torch
import torch.nn as nn
import torch.nn.functional as F

convs = {'1x1': dict(kernel_size=1, stride=1, padding=0, bias=False),
         '3x3': dict(kernel_size=3, stride=1, padding=1, bias=True),
         '5x5': dict(kernel_size=5, stride=1, padding=2, bias=True)}


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        channels_num = [64, 128, out_channels]  # according to the paper, out channels will be 256
        conv_down_params = dict(kernel_size=5, padding=2, stride=2, bias=False)
        conv_preserve_params = dict(kernel_size=5, stride=1, padding=2, bias=False)
        for i, channels in enumerate(channels_num):
            modules += [nn.Conv2d(in_channels=in_channels, out_channels=channels, **conv_preserve_params),
                        nn.BatchNorm2d(num_features=channels),
                        nn.ReLU()]
            modules += [nn.Conv2d(in_channels=channels, out_channels=channels, **conv_down_params),
                        nn.BatchNorm2d(num_features=channels),
                        nn.ReLU()]
            in_channels = channels
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        channels_num = [128, 64, out_channels]
        conv_up_params = dict(kernel_size=5, padding=2, stride=2, output_padding=1, bias=False)
        conv_preserve_params = dict(kernel_size=5, stride=1, padding=2, bias=False)
        for i, channels in enumerate(channels_num):
            modules += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=channels, **conv_up_params),
                        nn.BatchNorm2d(num_features=channels),
                        nn.ReLU()]
            modules += [nn.Conv2d(in_channels=channels, out_channels=channels, **conv_preserve_params)]

            if i != len(channels_num) - 1:
                modules += [nn.BatchNorm2d(num_features=channels),
                            nn.ReLU()]
            in_channels = channels
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        self.mu = nn.Linear(in_features=n_features, out_features=z_dim, bias=True)
        self.log_sigma2 = nn.Linear(in_features=n_features, out_features=z_dim, bias=True)
        self.z_to_h = nn.Linear(in_features=z_dim, out_features=n_features, bias=True)

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        device = next(self.parameters()).device
        features = self.features_encoder(x)
        features = features.reshape(x.shape[0], -1)
        mu = self.mu(features)
        log_sigma2 = self.log_sigma2(features)
        z = mu + torch.randn_like(mu, device=device) * torch.sqrt(torch.exp(log_sigma2))

        return z, mu, log_sigma2

    def decode(self, z):
        h = self.z_to_h(z)
        h = h.reshape(z.shape[0], *self.features_shape)
        x_rec = self.features_decoder(h)

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn((n, self.z_dim)).to(device)
            samples = self.decode(z)

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    N = x.shape[0]
    dz = z_mu.shape[1]
    data_loss = torch.mean((x - xr) ** 2) / x_sigma2
    kldiv_loss = (torch.sum(z_log_sigma2.exp()) + torch.sum(z_mu.pow(2)) - torch.sum(z_log_sigma2))/N - dz
    loss = data_loss + kldiv_loss
    return loss, data_loss, kldiv_loss