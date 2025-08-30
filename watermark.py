import numpy as np

# import pycircstat as pcs
from scipy.stats import norm


def rayleigh_test(angles):
    n = len(angles)
    R = np.sqrt(np.sum(np.cos(angles)) ** 2 + np.sum(np.sin(angles)) ** 2)
    z = (R**2) / n
    p_value = np.exp(-z)
    return z, p_value


def get_random_samples(dims, var=1 / 4):
    g = np.random.default_rng()
    return g.normal(0, var, size=dims)


def sample_unit_vector(secret_dim: int, seed=42) -> np.ndarray:
    g = np.random.default_rng(seed)
    secret_dir = g.normal(0, 1, size=secret_dim)
    return secret_dir / np.linalg.norm(secret_dir)


def vslice(start, step):
    return tuple(slice(x, x + y) for x, y in zip(start, step))


def inc_index(index, block_dim, shape):
    for i in reversed(range(len(index))):
        index[i] += block_dim[i]
        if index[i] < shape[i]:
            return index
        index[i] = 0
    return None


def pad_ones(l, dim):
    return (1,) * (l - len(dim)) + dim


def split_blocks(ar, block_dim):
    if len(block_dim) > ar.ndim:
        raise ValueError("block has more dimensions than array")
    block_dim = pad_ones(ar.ndim, block_dim)
    for i in range(ar.ndim):
        print(ar.shape[i], block_dim[i])
        if ar.shape[i] % block_dim[i] != 0:
            raise ValueError("Block dim does not divide array shape.")

    index = np.zeros(ar.ndim, dtype=int)
    while index is not None:
        yield ar[vslice(index, block_dim)]
        index = inc_index(index, block_dim, ar.shape)


def extract_blocks(ar, block_dim):
    return np.stack([b.flatten() for b in split_blocks(ar, block_dim)])


def restack_blocks(blocks, block_dim, shape):
    if len(block_dim) > len(shape):
        raise ValueError("block has more dimensions than array")
    block_dim = pad_ones(len(shape), block_dim)
    for i in range(len(shape)):
        if shape[i] % block_dim[i] != 0:
            raise ValueError("Block dim does not divide array shape.")
    ar = np.empty(shape, dtype=blocks.dtype)
    index = np.zeros(ar.ndim, dtype=int)
    i = 0
    while index is not None:
        ar[vslice(index, block_dim)] = blocks[i].reshape(block_dim)
        index = inc_index(index, block_dim, shape)
        i += 1
    return ar


def inner_prod_with_secret(samples, secret_direction):
    return extract_blocks(samples, secret_direction.shape) @ secret_direction.flatten()


def project_to_clwe(
    samples: np.ndarray, secret_direction: np.ndarray, gamma: float, beta=0
) -> np.ndarray:
    gammap = np.sqrt(beta * beta + gamma * gamma)
    inner_prod = inner_prod_with_secret(samples, secret_direction)
    k = np.round(gammap * inner_prod)
    errors = k * gamma / gammap
    if beta > 0:
        errors += get_random_samples(errors.shape, var=beta)
    errors = (errors / gammap) - inner_prod
    deltas = errors.reshape(-1, 1) @ secret_direction.reshape(1, -1)
    return samples + restack_blocks(deltas, secret_direction.shape, samples.shape)


def get_hclwe_errors(samples, secret_direction, gamma):
    inner_prod = inner_prod_with_secret(samples, secret_direction)
    return (gamma * inner_prod) % 1


def get_hclwe_score(samples, secret_direction, gamma):
    return rayleigh_test(
        2 * np.pi * get_hclwe_errors(samples, secret_direction, gamma)
    )[0]
    # return pcs.tests.rayleigh(2 * np.pi * get_hclwe_errors(samples, secret_direction, gamma))[0]


class CLWEWatermarker:
    def __init__(self, secret_dim, gamma, beta, seed) -> None:
        self.secret_dim = secret_dim
        self.gamma = gamma
        self.beta = beta
        self.seed = seed
        self.secret = sample_unit_vector(self.secret_dim, self.seed)

    def inject_watermark(self, latents_np: np.ndarray) -> np.ndarray:
        return project_to_clwe(latents_np, self.secret, self.gamma, self.beta)

    def get_errors(self, latents_np: np.ndarray) -> np.ndarray:
        return get_hclwe_errors(latents_np, self.secret, self.gamma)

    def get_score(self, latents_np: np.ndarray) -> float:
        return get_hclwe_score(latents_np, self.secret, self.gamma)
