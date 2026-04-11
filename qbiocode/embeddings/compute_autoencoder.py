import torch
import torch.nn as nn
import torch.optim as optim


# Define the Autoencoder Model
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, stride=2, padding=1),  # (64, 192, 192)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 96, 96)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (256, 48, 48)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (512, 24, 24)
            nn.ReLU(),
            nn.Conv2d(512, 7, kernel_size=3, stride=2, padding=1),  # (7, 16, 16)
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                7, 512, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (512, 24, 24)
            nn.ReLU(),
            nn.ConvTranspose2d(
                512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (256, 48, 48)
            nn.ReLU(),
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (128, 96, 96)
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (64, 192, 192)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 7, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (7, 384, 384)
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
