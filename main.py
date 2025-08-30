from diffusers import AutoencoderKL
import torch
from PIL import Image
import numpy as np
import sys
from watermark import CLWEWatermarker

device = "cpu"

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)

watermarker = CLWEWatermarker(8, 0.5, 0.001, 42)

def image_to_latent(image: Image.Image) -> torch.Tensor:
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    image_tensor = 2.0 * image_tensor - 1.0  # Normalize to [-1, 1]
    image_tensor = image_tensor.to("cpu")
    
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample() #* 0.18215
    return latent


def latent_to_image(latent: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        recon = vae.decode(latent).sample  # Undo scaling

    # Denormalize to [0, 1] and convert to image
    recon = (recon / 2 + 0.5).clamp(0, 1)
    recon_image = recon.cpu().permute(0, 2, 3, 1).numpy()[0]
    recon_image = (recon_image * 255).astype(np.uint8)
    return Image.fromarray(recon_image)

if len(sys.argv) > 1:
    image = Image.open(sys.argv[1]).convert("RGB")
else:
    # let image be a 512x512 white image
    image = Image.new('RGB', (512, 512), color = 'white')

latent = image_to_latent(image)
# new_latent = latent + 0.05 * (torch.randn_like(latent) * 2 - 1) 
new_latent = torch.tensor(watermarker.inject_watermark(latent.numpy(force=True)), 
                          device=latent.device, dtype=latent.dtype)

new_image = latent_to_image(new_latent)
new_image.show()

# recovered_latent = image_to_latent(new_image)
# print(torch.abs(new_latent - latent).mean())
# print(torch.abs(recovered_latent - new_latent).mean())
