# Watermarking Latent Representations via CLWE

This project is a proof of concept for watermarking arbitrary input images using their latent representations. The security of the watermarking scheme is based on the security of worst case lattice problems, particularly through the  Continuous Learning With Errors problem. This project is inspired by the recent work of [Shehata et. al.](https://arxiv.org/abs/2411.11434)

Authors: 

1. [Aditya Morolia](https://thecharmingsociopath.github.io)
2. [Yaonan Zhang]()

## Applications

## Technical Overview

### Adding a Watermark

### Testing for the presense of the watermark, given the secret key 

The Rayleigh test is a statistical method used to detect non-uniformity in circular data (angles). In this project, it is used to measure the detectability of the watermark signal embedded in latent vectors.

- **Null hypothesis ($H_0$):** The data is uniformly distributed (no preferred direction).
- **Alternative hypothesis ($H_1$):** The data shows a preferred direction (watermark present).

**How to Compute the Rayleigh score.**

1. **Compute the resultant vector length (`R`)** from the sum of unit vectors for each angle.
2. **Calculate the test statistic:**  
   `z = R² / n`  
   where `n` is the number of samples.
3. **Compute the p-value:**  
   `p = exp(-z)`  
   A small p-value (e.g., < 0.05) indicates the presence of a preferred direction (watermark detected).

**In This Project**,

- The Rayleigh score quantifies how much the watermark signal stands out from noise.
- A higher score (and lower p-value) means the watermark is more detectable.
- The test is used to evaluate the effectiveness of watermark injection and recovery.

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```
   python main.py path_to_image.png
   ```
   If no image is provided, a blank 512x512 image is used.

## References

- [CLUE-MARK: Watermarking Diffusion Models using CLWE](https://arxiv.org/abs/2411.11434)
