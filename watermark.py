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

### Takes a tensor of arbitary shape and converts it to chunks of size chunk_size
def latent_to_blocks(latents: torch.Tensor, block_size: int) -> torch.Tensor:
    latents_flat = latents.flatten()  # Flatten the tensor into a 1D array
    remainder = latents_flat.size(0) % block_size  # Check if padding is needed
    if remainder != 0:
        padding_size = block_size - remainder
        latents_flat = torch.cat([latents_flat, torch.zeros(padding_size, device=latents.device)], dim=0)
    latents_blocks = latents_flat.view(-1, block_size)  # Reshape into chunks of size chunk_size
    return latents_blocks

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

def generate_watermarked_latents(latents: torch.Tensor, secret: str, gamma: float, beta: float) -> torch.Tensor:
    ### Sample a gaussian random unit vector of dimension secret_dim
    u = sample_gaussian_unit_vector(secret_dim, seed=secret)
    latents_blocks = latent_to_blocks(latents, secret_dim)
    
    permutation = torch.randperm(latents_blocks.size(0), device=latents_blocks.device)  # Generate a random permutation
    inverse_permutation = torch.argsort(permutation)  # Store the inverse permutation for unshuffling later
    
    latents_blocks = latents_blocks[permutation]  # Shuffle the blocks
    project_to_clwe(latents_blocks, torch.tensor(u, device=latents_blocks.device), gamma, beta)
    