import os

# Project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for data and models
DATA_DIR = os.path.join(BASE_DIR, "synthetic_data", "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pt")

# Logging and Debugging
LOGGING_LEVEL = "INFO"
DEBUG_MODE = True

# API Configuration
API_URL = "https://your-api-endpoint.com"
API_KEY = os.getenv("API_KEY", "your_default_api_key_here")

# Dashboard settings
DASHBOARD_REFRESH_RATE = 1  # Refresh rate in seconds

# Model Parameters for VAE-GAN
VAE_GAN_PARAMS = {
    "latent_dim": 128,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 50
}

# Function to retrieve configuration as a dictionary
def get_config():
    """Returns the configuration settings as a dictionary."""
    return {
        "BASE_DIR": BASE_DIR,
        "DATA_DIR": DATA_DIR,
        "MODEL_PATH": MODEL_PATH,
        "LOGGING_LEVEL": LOGGING_LEVEL,
        "DEBUG_MODE": DEBUG_MODE,
        "API_URL": API_URL,
        "API_KEY": API_KEY,
        "DASHBOARD_REFRESH_RATE": DASHBOARD_REFRESH_RATE,
        "VAE_GAN_PARAMS": VAE_GAN_PARAMS,
    }
