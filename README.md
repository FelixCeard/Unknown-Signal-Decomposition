# Unknown-Signal-Decomposition

The idea is to be able to deconstruct a Song into its individual parts.
To this end, we would train a sort of VAE.

## Architecture

[//]: # (Ideally, a sort of U-Net would be great, our dear facebook-freedom researcher have created a transformer encoder/decoder &#40;[link]&#40;https://github.com/facebookresearch/demucs&#41;&#41;.)
The current idea is to transform the audio file to freuencies, as it produces an image.
Afterwards, I would train a VQ-VAE, to encode the data into latent-space, and finally decode the data conditionally with
a transformer. In short, just like in the `Taming Transformers` repo.

## Training

Assuming that a song is constructed like this:
![idea.svg](images%2Fidea.svg)

The training of the VAE would be preformed in 2 steps:

**Step 1**:
![VAE-1.step.svg](images%2FVAE-1.step.svg)

**Step 2**:
![VAE-2.svg](images%2FVAE-2.svg)

## Inference

Just like Step 2 of training

## Progress
- VAE has 4 input channels:
  - Left Real
  - Left Imag
  - Right Real
  - Right Imag
- Audio encoding is via normal FFT transformation, therefore accounting for (almost) lossless encoding (good enough for this project)

