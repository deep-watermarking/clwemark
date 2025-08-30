import numpy as np
import torch

secret_dim = 64  ## Dimension of secret
gamma = 0.5 ## Separation between pancakes
beta = 0.001 ## Variance of each pancake

def sample_gaussian_unit_vector(dim, seed=42):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    unit_vec = vec / np.linalg.norm(vec)
    return unit_vec

### TODO: check for remainder when chunking
### Takes a tensor of arbitary shape and converts it to chunks of size chunk_size
def latent_to_blocks(latents: torch.Tensor, chunk_size: int) -> torch.Tensor:
    latents_flat = latents.flatten()  # Flatten the tensor into a 1D array
    latents_chunks = latents_flat.view(-1, secret_dim)  # Reshape into chunks of size secret_dim
    return latents_chunks

### TODO: Test this function
def blocks_to_latent(blocks: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    latents_flat = blocks.view(-1)  # Flatten the blocks back to 1D
    latents = latents_flat.view(original_shape)  # Reshape back to the original shape
    return latents

def compute_blockwise_innerproducts(latents: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    inner_products = torch.matmul(latent_to_blocks(latents), u)  # Compute inner product of each chunk with u
    return inner_products

### TODO
def project_to_clwe(latents: torch.Tensor, u: torch.Tensor, gamma: float, beta: float) -> torch.Tensor:
    pass

def generate_watermarked_latents(latents: torch.Tensor, secret: str) -> torch.Tensor:
    ### Sample a gaussian random unit vector of dimension secret_dim
    u = sample_gaussian_unit_vector(secret_dim, seed=secret)
