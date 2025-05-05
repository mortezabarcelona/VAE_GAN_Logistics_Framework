# File: src/models/optimization/vae_gan.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ========== VAE Components ==========
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, z):
        return self.fc(z)


# ========== GAN Discriminator ==========
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


# ========== Integrated VAE-GAN Model ==========
class VAEGAN(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEGAN, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # VAE branch: encode and reconstruct
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        # GAN branch: pass reconstructed sample through the discriminator
        disc_out = self.discriminator(recon_x)
        return recon_x, mu, logvar, disc_out


# ========== Hybrid Loss Function ==========
def vaegan_loss(recon_x, x, mu, logvar, disc_out):
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL Divergence Loss for latent regularization
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Adversarial loss (BCE Loss on discriminator's output)
    adversarial_loss = nn.BCELoss()(disc_out, torch.ones_like(disc_out))
    return recon_loss + kl_loss + adversarial_loss


# ========== Training Loop ==========
def train_vaegan(train_loader, input_dim, latent_dim=8, epochs=20, lr=1e-3, device="cpu"):
    model = VAEGAN(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar, disc_out = model(data)
            loss = vaegan_loss(recon_x, data, mu, logvar, disc_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
    return model


# ========== Data Loading and Training Example ==========
if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Construct the absolute path to the cleaned synthetic data file.
    data_file = os.path.abspath(
        os.path.join("..", "..", "synthetic_data", "data", "processed", "synthetic_logistics_data_cleaned.csv"))
    print("Loading data from:", data_file)

    # Load the cleaned data.
    df = pd.read_csv(data_file)

    # Select the relevant features for training.
    features = ['volume', 'cost', 'transit_time', 'co2_emissions', 'cost_per_tonkkm']
    df_model = df[features].dropna()  # Ensure there are no missing values.

    # Normalize features using StandardScaler.
    scaler = StandardScaler()
    data_array = scaler.fit_transform(df_model.values)

    # Convert the NumPy array to a PyTorch tensor.
    data_tensor = torch.tensor(data_array, dtype=torch.float32)

    # Create a DataLoader from the tensor.
    dataset = TensorDataset(data_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Determine if a GPU is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the integrated VAE-GAN model on your synthetic logistics data.
    trained_model = train_vaegan(train_loader, input_dim=5, latent_dim=8, epochs=20, lr=1e-3, device=device)

    # Save the trained model to disk for later evaluation.
    model_save_path = os.path.abspath(os.path.join("..", "..", "models", "trained_model.pt"))
    torch.save(trained_model.state_dict(), model_save_path)

    print("Trained model saved to:", model_save_path)
