from diffusers import AutoencoderKL
import torch
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
from watermark import CLWEWatermarker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae"
).to(DEVICE)

watermarker = CLWEWatermarker(8, 0.5, 0.001, 42)


def image_to_latent(image: Image.Image) -> torch.Tensor:
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    image_tensor = 2.0 * image_tensor - 1.0  # Normalize to [-1, 1]
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()  # * 0.18215
    return latent


def latent_to_image(latent: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        recon = vae.decode(latent).sample  # Undo scaling

    # Denormalize to [0, 1] and convert to image
    recon = (recon / 2 + 0.5).clamp(0, 1)
    recon_image = recon.cpu().permute(0, 2, 3, 1).numpy()[0]
    recon_image = (recon_image * 255).astype(np.uint8)
    return Image.fromarray(recon_image)


def draw_distribution(latent: torch.Tensor):
    latent_np = latent.cpu().numpy()  # shape: (1, C, H, W)
    latent_flat = latent_np.reshape(latent_np.shape[1], -1)  # shape: (C, H*W)

    print(latent_flat.shape)

    for i in range(latent_flat.shape[0]):  # plot first 4 channels
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(latent_flat[i], bins=100, density=True)
        plt.title(f"Histogram of latent channel {i}")
        plt.subplot(1, 2, 2)
        stats.probplot(latent_flat[i], dist="norm", plot=plt)
        plt.title(f"Q-Q plot of latent channel {i}")
        plt.show()


if len(sys.argv) > 1:
    image = Image.open(sys.argv[1]).convert("RGB")
else:
    # let image be a 512x512 white image
    image = Image.new("RGB", (512, 512), color="white")

latent = image_to_latent(image)

watermarked_latent = torch.tensor(
    watermarker.inject_watermark(latent.numpy(force=True)),
    device=latent.device,
    dtype=latent.dtype,
)

watermarked_image = latent_to_image(watermarked_latent)
watermarked_image.show()

recovered_latent = image_to_latent(watermarked_image)
print("Score - original latent:", watermarker.get_score(latent.numpy(force=True)))
print(
    "Score - watermarked latent:",
    watermarker.get_score(watermarked_latent.numpy(force=True)),
)
print(
    "Score - recovered latent:",
    watermarker.get_score(recovered_latent.numpy(force=True)),
)
