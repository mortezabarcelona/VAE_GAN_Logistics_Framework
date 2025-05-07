# %% [code]
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_mse(original, reconstructed):
    """Calculate Mean Squared Error (MSE)."""
    return mean_squared_error(original, reconstructed)


def compute_mae(original, reconstructed):
    """Calculate Mean Absolute Error (MAE)."""
    return mean_absolute_error(original, reconstructed)


def compute_ssim(original, reconstructed):
    """Calculate Structural Similarity Index (SSIM) for feature-based data."""
    ssim_scores = []
    feature_dim = original.shape[1]  # Get the number of features per sample
    win_size = min(7, feature_dim)  # Ensure win_size does not exceed feature count

    for i in range(original.shape[0]):
        ssim_scores.append(
            ssim(original[i], reconstructed[i], data_range=original.max() - original.min(), win_size=win_size))

    return np.mean(ssim_scores)


def evaluate_reconstruction(original_data, reconstructed_data):
    """Compute MSE, MAE, and SSIM for reconstruction evaluation."""
    original_np = original_data.cpu().numpy()
    reconstructed_np = reconstructed_data.cpu().numpy()

    mse_score = compute_mse(original_np, reconstructed_np)
    mae_score = compute_mae(original_np, reconstructed_np)
    ssim_score = compute_ssim(original_np, reconstructed_np)

    return {"MSE": mse_score, "MAE": mae_score, "SSIM": ssim_score}


if __name__ == "__main__":
    # Example usage with a PyTorch tensor dataset (Replace with actual tensors)
    num_examples = 5
    sample_data = torch.tensor(np.random.rand(num_examples, 5), dtype=torch.float32)
    recon_data = sample_data + torch.tensor(np.random.normal(0, 0.1, sample_data.shape), dtype=torch.float32)

    results = evaluate_reconstruction(sample_data, recon_data)
    print("Reconstruction Error Metrics:", results)
