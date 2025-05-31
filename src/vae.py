import torch
import torch.nn.functional as F


def calculate_mse_reconstruction_loss(original_data, reconstructed_data):
    """
    Calculates the Mean Squared Error (MSE) for the reconstruction loss.
    This measures how well the VAE's decoder reconstructs the original input.

    Args:
        original_data (torch.Tensor): The original input data (e.g., image pixels).
                                     Shape: (batch_size, data_dim) or (batch_size, channels, height, width)
        reconstructed_data (torch.Tensor): The data reconstructed by the decoder.
                                          Shape: (batch_size, data_dim) or (batch_size, channels, height, width)

    Returns:
        torch.Tensor: The scalar MSE reconstruction loss.
    """
    # Flatten the tensors if they are multi-dimensional (e.g., images)
    # This ensures MSE is calculated across all elements
    original_data_flat = original_data.view(original_data.size(0), -1)
    reconstructed_data_flat = reconstructed_data.view(reconstructed_data.size(0), -1)

    # Calculate MSE loss
    # reduction='sum' sums the loss over all elements in the batch
    # reduction='mean' would average it over the batch
    mse_loss = F.mse_loss(reconstructed_data_flat, original_data_flat, reduction="sum")
    return mse_loss


def calculate_kl_divergence_loss(mu, log_var):
    """
    Calculates the Kullback-Leibler (KL) Divergence loss.
    This term regularizes the latent space, pushing the learned distribution
    towards a standard normal distribution (mean=0, variance=1).

    The formula for KL divergence between a Gaussian N(mu, sigma^2) and
    a standard Gaussian N(0, 1) is:
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu (torch.Tensor): The mean (mu) of the latent distribution predicted by the encoder.
                           Shape: (batch_size, latent_dim)
        log_var (torch.Tensor): The log-variance (log(sigma^2)) of the latent distribution.
                                Shape: (batch_size, latent_dim)

    Returns:
        torch.Tensor: The scalar KL divergence loss.
    """
    # Ensure log_var is not negative (though it should be handled by the network output)
    # For numerical stability, sometimes sigma^2 is calculated as exp(log_var)
    # sigma_squared = torch.exp(log_var)

    # KL divergence formula
    # The sum is over the latent dimensions, and the mean is over the batch
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl_loss


# --- Example Usage ---
if __name__ == "__main__":
    # Simulate a batch of original and reconstructed data (e.g., 10 samples, 784 features each)
    batch_size = 10
    data_dimension = 784  # e.g., 28x28 image flattened

    original_data_example = torch.randn(batch_size, data_dimension)
    reconstructed_data_example = (
        torch.randn(batch_size, data_dimension) * 0.9 + original_data_example * 0.1
    )  # Make it somewhat similar

    # Simulate encoder outputs for mu and log_var
    latent_dimension = 20
    mu_example = torch.randn(batch_size, latent_dimension)
    log_var_example = (
        torch.randn(batch_size, latent_dimension) * 0.5
    )  # Smaller variance

    # Calculate Reconstruction Loss
    reconstruction_loss = calculate_mse_reconstruction_loss(
        original_data_example, reconstructed_data_example
    )
    print(f"Reconstruction Loss (MSE): {reconstruction_loss.item():.4f}")

    # Calculate KL Divergence Loss
    kl_divergence_loss = calculate_kl_divergence_loss(mu_example, log_var_example)
    print(f"KL Divergence Loss: {kl_divergence_loss.item():.4f}")

    # Total VAE Loss (simplified, usually averaged over batch)
    total_vae_loss = reconstruction_loss + kl_divergence_loss
    print(f"Total VAE Loss (Reconstruction + KL): {total_vae_loss.item():.4f}")

    # A more common way to combine and average the loss
    # The total loss is typically averaged over the batch size
    total_vae_loss_per_sample = (reconstruction_loss + kl_divergence_loss) / batch_size
    print(f"Total VAE Loss per sample: {total_vae_loss_per_sample.item():.4f}")
