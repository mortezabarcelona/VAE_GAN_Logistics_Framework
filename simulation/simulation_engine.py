import sys
import os
import torch
import numpy as np

# Add the project root to sys.path for module resolution.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the VAE-GAN model.
from src.models.optimization.vae_gan import VAEGAN

# Set device to use GPU if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, input_dim=5, latent_dim=8):
    """
    Loads the pre-trained VAE-GAN model by reinitializing its architecture and loading
    the state dictionary.

    Parameters:
      model_path (str): Path to the saved state dictionary.
      input_dim (int): Number of features in the input.
      latent_dim (int): Dimension of the latent space.

    Returns:
      model (VAEGAN): The VAE-GAN model loaded to the specified device and in evaluation mode.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    # Initialize the model architecture.
    model = VAEGAN(input_dim=input_dim, latent_dim=latent_dim)

    # Load the state dictionary.
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move the model to the appropriate device and set to evaluation mode.
    model.to(device)
    model.eval()

    return model


def generate_synthetic_scenarios(model, input_data):
    """
    Given input logistics data, use the trained VAE-GAN to generate synthetic scenarios.

    This function returns two outputs:
      - reconstruction: The predicted/reconstructed logistics features.
      - latent_embedding: The latent representation using the encoder’s output (mu).

    Parameters:
      input_data: A numpy array or torch.Tensor of raw input features.

    Returns:
      reconstruction (np.array): Reconstructed forecasts.
      latent_embedding (np.array): Latent embeddings (using encoder's mu).
    """
    # Convert input_data to torch.Tensor if it's not one already.
    if not torch.is_tensor(input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    else:
        input_tensor = input_data.to(device)

    with torch.no_grad():
        # Forward pass through the model.
        # The model returns four outputs: reconstruction, mu, logvar, and disc_out.
        reconstruction, mu, logvar, _ = model(input_tensor)
        # Use the encoder's mu as the latent embedding.
        latent_embedding = mu

    # Return numpy arrays.
    return reconstruction.cpu().numpy(), latent_embedding.cpu().numpy()


if __name__ == "__main__":
    # Define the path to the model file.
    model_path = os.path.join(project_root, "src", "models", "trained_model.pt")
    print("Loading model from:", model_path)

    # Load the model.
    model = load_model(model_path, input_dim=5, latent_dim=8)

    # Example input data for simulation (replace with real or synthetic logistics features as needed).
    example_input = np.array([
        [0.5, -0.1, 1.2, 0.3, -0.2],
        [0.8, 0.0, 1.5, 0.4, -0.1]
    ])

    # Generate synthetic scenarios.
    reconstruction, latent_embedding = generate_synthetic_scenarios(model, example_input)

    print("Reconstructed Forecasts:")
    print(reconstruction)
    print("\nLatent Embeddings:")
    print(latent_embedding)
