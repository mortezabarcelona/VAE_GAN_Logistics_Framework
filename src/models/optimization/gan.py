# File: src/models/optimization/gan.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Assuming features are scaled; adjust as needed.
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_gan(generator, discriminator, data_loader, noise_dim, epochs=50, lr=1e-3, device='cpu'):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss_D = 0.0
        total_loss_G = 0.0
        for _, (real_data,) in enumerate(data_loader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # Create real and fake labels
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Discriminator loss on real data
            output_real = discriminator(real_data)
            loss_real = criterion(output_real, valid)

            # Generate fake data
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_data = generator(noise)
            output_fake = discriminator(fake_data.detach())
            loss_fake = criterion(output_fake, fake)

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            output_fake = discriminator(fake_data)
            loss_G = criterion(output_fake, valid)
            loss_G.backward()
            optimizer_G.step()

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()

        avg_loss_D = total_loss_D / len(data_loader)
        avg_loss_G = total_loss_G / len(data_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss_D: {avg_loss_D:.4f}, Loss_G: {avg_loss_G:.4f}")

    return generator, discriminator


if __name__ == '__main__':
    # For demonstration, create dummy data with 5 features (similar to your logistics features)
    import numpy as np

    np.random.seed(42)
    dummy_data = np.random.randn(1000, 5)  # 1000 samples, 5 features
    data_tensor = torch.tensor(dummy_data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_dim = 10  # Dimension of the noise vector for generator
    output_dim = 5  # Must match the feature dimension of real data
    generator = Generator(noise_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim).to(device)

    train_gan(generator, discriminator, data_loader, noise_dim, epochs=20, lr=1e-3, device=device)
