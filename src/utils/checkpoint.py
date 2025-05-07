import torch

def save_model(model, path="trained_model.pt"):
    """Save the trained model."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="trained_model.pt"):
    """Load a trained model."""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model
