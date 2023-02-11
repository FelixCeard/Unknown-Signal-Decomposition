# Unknown-Signal-Decomposition
The idea is to be able to deconstruct a Song into its individual parts. 
To this end, we would train a sort of VAE.

## Architecture
Ideally, a sort of U-Net would be great, our dear facebook-freedom researcher have created a transformer encoder/decoder ([link](https://github.com/facebookresearch/demucs)).

This idea A good start, I will try to fine-tune their model.


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

