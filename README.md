# Watermarking Latent Representations via CLWE

*AI for Cybersecurity, and Security for AI*

This project is a proof of concept for watermarking arbitrary input images using their latent representations. The security of the watermarking scheme is based on the security of worst case lattice problems, particularly through the  Continuous Learning With Errors problem.

This project was built for a TikTok hackathon under the theme **AI x Cybersecurity: use AI for cybersecurity, use cybersecurity for AI**. Our solution leverages the security of lattice-based cryptography (LWE) and the power of AI models to watermark arbitrary input images, not just generated ones. This enables platforms like TikTok to verify content authenticity and prevent reward payouts for stolen or reposted content, addressing both AI for security and security for AI.

This project is inspired by the recent work of [Shehata et. al.](https://arxiv.org/abs/2411.11434)

Authors:

1. [Aditya Morolia](https://thecharmingsociopath.github.io)
2. [Yaonan Zhang]()

---

## Problem Statement

**How can we use AI to enhance cybersecurity, and how can cybersecurity principles strengthen AI?**

Our idea:  
- **AI for Security:** Use AI models to embed robust, cryptographically secure watermarks into images, enabling content provenance and anti-piracy measures.
- **Security for AI:** Rely on the hardness of the Continuous Learning With Errors (CLWE) problem, a worst-case lattice problem, to ensure watermark security and resistance to removal or forgery.

---

## Features & Functionality

- **Watermarking Arbitrary Images:**  
  Any input image is encoded into a latent space using a pre-trained VAE (from Stable Diffusion). A CLWE-based watermark is injected into the latent representation.
- **Watermark Detection:**  
  Given the secret key, the watermark can be statistically detected using the Rayleigh test, which measures the presence of a preferred direction in the latent space.
- **Content Provenance:**  
  Platforms can verify if content is original or watermarked, preventing fraudulent reward payouts for stolen content.
- **Security Guarantees:**  
  The watermark’s security is based on the hardness of lattice problems, making it resistant to adversarial removal.

---

## Technical Overview

### Development Tools

- **Programming Language:** Python 3.10+
- **IDE:** Visual Studio Code (recommended)
- **Platform:** macOS (tested), should work on Linux/Windows

### APIs & Assets

- **Pre-trained Model:**  
  [Stable Diffusion VAE](https://huggingface.co/CompVis/stable-diffusion-v1-4) via HuggingFace Diffusers API
- **Image Assets:**  
  Any PNG/JPG image; sample images provided in the `images/` folder

### Libraries Used

- [`diffusers`](https://github.com/huggingface/diffusers) (for VAE encoding/decoding)
- [`torch`](https://pytorch.org/) (PyTorch, for tensor operations)
- [`torchvision`](https://pytorch.org/vision/stable/index.html) (image utilities)
- [`transformers`](https://github.com/huggingface/transformers) (model loading)
- [`matplotlib`](https://matplotlib.org/) (visualization)
- [`scipy`](https://scipy.org/) (statistics, Rayleigh test)
- [`numpy`](https://numpy.org/) (numerical operations)
- [`Pillow`](https://python-pillow.org/) (image manipulation)

### Watermarking Process

1. **Encoding:**  
   The input image is resized and encoded into a latent representation using the VAE.
2. **Watermark Injection:**  
   A secret direction (unit vector) is sampled using a Gaussian distribution. The latent is perturbed along this direction using CLWE principles.
3. **Decoding:**  
   The watermarked latent is decoded back into an image.
4. **Detection:**  
   The Rayleigh test is applied to the projected errors in the latent space to statistically detect the watermark.

### Testing for the presense of the watermark, given the secret key

The Rayleigh test is a statistical method used to detect non-uniformity in circular data (angles). In this project, it is used to measure the detectability of the watermark signal embedded in latent vectors.

- **Null hypothesis ($H_0$):** The data is uniformly distributed (no preferred direction).
- **Alternative hypothesis ($H_1$):** The data shows a preferred direction (watermark present).

**How to Compute the Rayleigh score.**

1. **Compute the resultant vector length $R$** from the sum of unit vectors for each angle.
2. **Calculate the test statistic:**  
   $z = R^2 / n$  
   where $n$ is the number of samples.
3. **Compute the $p$-value:**  
   $p = exp(-z)$
   A small $p$-value (e.g., $< 0.05$) indicates the presence of a preferred direction (watermark detected).

**In This Project**,

- The Rayleigh score quantifies how much the watermark signal stands out from noise.
- A higher score (and lower p-value) means the watermark is more detectable.
- The test is used to evaluate the effectiveness of watermark injection and recovery.

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py path_to_image.png`

If no image is provided, a blank 512x512 image is used.

## Applications

- **Content Provenance:**  
  Platforms like TikTok can verify the originality of uploaded images and prevent reward payouts for stolen content.
- **Anti-Piracy:**  
  Robust watermarking makes it difficult for adversaries to remove or forge watermarks.
- **AI Model Security:**  
  Demonstrates how cryptographic principles can be used to secure AI-generated and AI-processed content.

## Future Work

- Extend watermarking to video and audio content.
- Improve robustness against adversarial attacks and image transformations.
- Integrate with content management systems for automated provenance checks.
- Explore alternative statistical tests and watermarking schemes.


## References

- [CLUE-MARK: Watermarking Diffusion Models using CLWE](https://arxiv.org/abs/2411.11434)
- [Stable Diffusion VAE](https://huggingface.co/CompVis/stable-diffusion-v1-4)
