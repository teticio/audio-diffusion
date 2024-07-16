---
title: Audio Diffusion
emoji: 🎵
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 3.1.4
app_file: app.py
pinned: false
license: gpl-3.0
---
# audio-diffusion [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teticio/audio-diffusion/blob/master/notebooks/gradio_app.ipynb)

## Apply diffusion models to synthesize music instead of images using the new Hugging Face [diffusers](https://github.com/huggingface/diffusers) package

---
#### Sample automatically generated loop

https://user-images.githubusercontent.com/44233095/204103172-27f25d63-5e77-40ca-91ab-d04a45d4726f.mp4

Go to https://soundcloud.com/teticio2/sets/audio-diffusion-loops for more examples.

---
#### Updates

**25/12/2022**. Now it is possible to train models conditional on an encoding (of text or audio, for example). See the section on Conditional Audio Generation below.

**5/12/2022**. 🤗 Exciting news! [`AudioDiffusionPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audio_diffusion) has been migrated to the Hugging Face `diffusers` package so that it is even easier for others to use and contribute.

**2/12/2022**. Added Mel to pipeline and updated the pretrained models to save Mel config (they are now no longer compatible with previous versions of this repo). It is relatively straightforward to migrate previously trained models to the new format (see https://huggingface.co/teticio/audio-diffusion-256).

**7/11/2022**. Added pre-trained latent audio diffusion models [teticio/latent-audio-diffusion-256](https://huggingface.co/teticio/latent-audio-diffusion-256) and [teticio/latent-audio-diffusion-ddim-256](https://huggingface.co/teticio/latent-audio-diffusion-ddim-256). You can use the pre-trained VAE to train your own latent diffusion models on a different set of audio files.

**22/10/2022**. Added DDIM encoder and ability to interpolate between audios in latent "noise" space. Mel spectrograms no longer have to be square (thanks to Tristan for this one), so you can set the vertical (frequency) and horizontal (time) resolutions independently.

**15/10/2022**. Added latent audio diffusion (see below). Also added the possibility to train a DDIM ([De-noising Diffusion Implicit Models](https://arxiv.org/pdf/2010.02502.pdf)). These have the benefit that samples can be generated with much fewer steps (~50) than used in training.

**4/10/2022**. It is now possible to mask parts of the input audio during generation which means you can stitch several samples together (think "out-painting").

**27/9/2022**. You can now generate an audio based on a previous one. You can use this to generate variations of the same audio or even to "remix" a track (via a sort of "style transfer"). You can find examples of how to do this in the [`test_model.ipynb`](https://colab.research.google.com/github/teticio/audio-diffusion/blob/master/notebooks/test_model.ipynb) notebook.

---

![mel spectrogram](https://user-images.githubusercontent.com/44233095/205305826-8b39c917-26c5-49b4-887c-776f5d69e970.png)

---

## DDPM ([De-noising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239))

Audio can be represented as images by transforming to a [mel spectrogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), such as the one shown above. The class `Mel` in `mel.py` can convert a slice of audio into a mel spectrogram of `x_res` x `y_res` and vice versa. The higher the resolution, the less audio information will be lost. You can see how this works in the [`test_mel.ipynb`](https://github.com/teticio/audio-diffusion/blob/main/notebooks/test_mel.ipynb) notebook.

A DDPM is trained on a set of mel spectrograms that have been generated from a directory of audio files. It is then used to synthesize similar mel spectrograms, which are then converted back into audio.

You can play around with some pre-trained models on [Google Colab](https://colab.research.google.com/github/teticio/audio-diffusion/blob/master/notebooks/test_model.ipynb) or [Hugging Face spaces](https://huggingface.co/spaces/teticio/audio-diffusion). Check out some automatically generated loops [here](https://soundcloud.com/teticio2/sets/audio-diffusion-loops).

| Model | Dataset | Description |
|-------|---------|-------------|
| [teticio/audio-diffusion-256](https://huggingface.co/teticio/audio-diffusion-256) | [teticio/audio-diffusion-256](https://huggingface.co/datasets/teticio/audio-diffusion-256) | My "liked" Spotify playlist |
| [teticio/audio-diffusion-breaks-256](https://huggingface.co/teticio/audio-diffusion-breaks-256) | [teticio/audio-diffusion-breaks-256](https://huggingface.co/datasets/teticio/audio-diffusion-breaks-256) | Samples that have been used in music, sourced from [WhoSampled](https://whosampled.com) and [YouTube](https://youtube.com) |
| [teticio/audio-diffusion-instrumental-hiphop-256](https://huggingface.co/teticio/audio-diffusion-instrumental-hiphop-256) | [teticio/audio-diffusion-instrumental-hiphop-256](https://huggingface.co/datasets/teticio/audio-diffusion-instrumental-hiphop-256) | Instrumental Hip Hop music |
| [teticio/audio-diffusion-ddim-256](https://huggingface.co/teticio/audio-diffusion-ddim-256) | [teticio/audio-diffusion-256](https://huggingface.co/datasets/teticio/audio-diffusion-256) | De-noising Diffusion Implicit Model |
| [teticio/latent-audio-diffusion-256](https://huggingface.co/teticio/latent-audio-diffusion-256) | [teticio/audio-diffusion-256](https://huggingface.co/datasets/teticio/audio-diffusion-256) | Latent Audio Diffusion model |
| [teticio/latent-audio-diffusion-ddim-256](https://huggingface.co/teticio/latent-audio-diffusion-ddim-256) | [teticio/audio-diffusion-256](https://huggingface.co/datasets/teticio/audio-diffusion-256) | Latent Audio Diffusion Implicit Model |
| [teticio/conditional-latent-audio-diffusion-512](https://huggingface.co/teticio/latent-audio-diffusion-512) | [teticio/audio-diffusion-512](https://huggingface.co/datasets/teticio/audio-diffusion-512) | Conditional Latent Audio Diffusion Model |

---

## Generate Mel spectrogram dataset from directory of audio files

#### Install from GitHub (includes training scripts)

```bash
git clone https://github.com/teticio/audio-diffusion.git
cd audio-diffusion
pip install .
```

#### Install from PyPI

```bash
pip install audiodiffusion
```

#### Training can be run with Mel spectrograms of resolution 64x64 on a single commercial grade GPU (e.g. RTX 2080 Ti). The `hop_length` should be set to 1024 for better results

```bash
python scripts/audio_to_images.py \
--resolution 64,64 \
--hop_length 1024 \
--input_dir path-to-audio-files \
--output_dir path-to-output-data
```

#### Generate dataset of 256x256 Mel spectrograms and push to hub (you will need to be authenticated with `huggingface-cli login`)

```bash
python scripts/audio_to_images.py \
--resolution 256 \
--input_dir path-to-audio-files \
--output_dir data/audio-diffusion-256 \
--push_to_hub teticio/audio-diffusion-256
```

Note that the default `sample_rate` is 22050 and audios will be resampled if they are at a different rate. If you change this value, you may find that the results in the `test_mel.ipynb` notebook are not good (for example, if `sample_rate` is 48000) and that it is necessary to adjust `n_fft` (for example, to 2000 instead of the default value of 2048; alternatively, you can resample to a `sample_rate` of 44100). Make sure you use the same parameters for training and inference. You should also bear in mind that not all resolutions work with the neural network architecture as currently configured - you should be safe if you stick to powers of 2.

## Train model

#### Run training on local machine

```bash
accelerate launch --config_file config/accelerate_local.yaml \
scripts/train_unet.py \
--dataset_name data/audio-diffusion-64 \
--hop_length 1024 \
--output_dir models/ddpm-ema-audio-64 \
--train_batch_size 16 \
--num_epochs 100 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-4 \
--lr_warmup_steps 500 \
--mixed_precision no
```

#### Run training on local machine with `batch_size` of 2 and `gradient_accumulation_steps` 8 to compensate, so that 256x256 resolution model fits on commercial grade GPU and push to hub

```bash
accelerate launch --config_file config/accelerate_local.yaml \
scripts/train_unet.py \
--dataset_name teticio/audio-diffusion-256 \
--output_dir models/audio-diffusion-256 \
--num_epochs 100 \
--train_batch_size 2 \
--eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--learning_rate 1e-4 \
--lr_warmup_steps 500 \
--mixed_precision no \
--push_to_hub True \
--hub_model_id audio-diffusion-256 \
--hub_token $(cat $HOME/.huggingface/token)
```

#### Run training on SageMaker

```bash
accelerate launch --config_file config/accelerate_sagemaker.yaml \
scripts/train_unet.py \
--dataset_name teticio/audio-diffusion-256 \
--output_dir models/ddpm-ema-audio-256 \
--train_batch_size 16 \
--num_epochs 100 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-4 \
--lr_warmup_steps 500 \
--mixed_precision no
```

## DDIM ([De-noising Diffusion Implicit Models](https://arxiv.org/pdf/2010.02502.pdf))

#### A DDIM can be trained by adding the parameter

```bash
--scheduler ddim
```

Inference can the be run with far fewer steps than the number used for training (e.g., ~50), allowing for much faster generation. Without retraining, the parameter `eta` can be used to replicate a DDPM if it is set to 1 or a DDIM if it is set to 0, with all values in between being valid. When `eta` is 0 (the default value), the de-noising procedure is deterministic, which means that it can be run in reverse as a kind of encoder that recovers the original noise used in generation. A function `encode` has been added to `AudioDiffusionPipeline` for this purpose. It is then possible to interpolate between audios in the latent "noise" space using the function `slerp` (Spherical Linear intERPolation).

## Latent Audio Diffusion

Rather than de-noising images directly, it is interesting to work in the "latent space" after first encoding images using an autoencoder. This has a number of advantages. Firstly, the information in the images is compressed into a latent space of a much lower dimension, so it is much faster to train de-noising diffusion models and run inference with them. Secondly, similar images tend to be clustered together and interpolating between two images in latent space can produce meaningful combinations.

At the time of writing, the Hugging Face `diffusers` library is geared towards inference and lacking in training functionality (rather like its cousin `transformers` in the early days of development). In order to train a VAE (Variational AutoEncoder), I use the [stable-diffusion](https://github.com/CompVis/stable-diffusion) repo from CompVis and convert the checkpoints to `diffusers` format. Note that it uses a perceptual loss function for images; it would be nice to try a perceptual *audio* loss function.

#### Train latent diffusion model using pre-trained VAE

```bash
accelerate launch ...
...
--vae teticio/latent-audio-diffusion-256
```

#### Install dependencies to train with Stable Diffusion

```bash
pip install omegaconf pytorch_lightning==1.7.7 torchvision einops
pip install -e git+https://github.com/CompVis/stable-diffusion.git@main#egg=latent-diffusion
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
```

#### Train an autoencoder

```bash
python scripts/train_vae.py \
--dataset_name teticio/audio-diffusion-256 \
--batch_size 2 \
--gradient_accumulation_steps 12
```

#### Train latent diffusion model

```bash
accelerate launch ...
...
--vae models/autoencoder-kl
```

## Conditional Audio Generation

We can generate audio conditional on a text prompt - or indeed anything which can be encoded into a bunch of numbers - much like DALL-E2, Midjourney and Stable Diffusion. It is generally harder to find good quality datasets of audios together with descriptions, although the people behind the dataset used to train Stable Diffusion are making some very interesting progress [here](https://github.com/LAION-AI/audio-dataset). I have chosen to encode the audio directly instead based on "how it sounds", using a [model which I trained on hundreds of thousands of Spotify playlists](https://github.com/teticio/Deej-AI). To encode an audio into a 100 dimensional vector

```python
from audiodiffusion.audio_encoder import AudioEncoder

audio_encoder = AudioEncoder.from_pretrained("teticio/audio-encoder")
audio_encoder.encode(['/home/teticio/Music/liked/Agua Re - Holy Dance - Large Sound Mix.mp3'])
```

Once you have prepared a dataset, you can encode the audio files with this script

```bash
python scripts/encode_audio \
--dataset_name teticio/audio-diffusion-256 \
--out_file data/encodings.p
```

Then you can train a model with

```bash
accelerate launch ...
...
--encodings data/encodings.p
```

When generating audios, you will need to pass an `encodings` Tensor. See the [`conditional_generation.ipynb`](https://colab.research.google.com/github/teticio/audio-diffusion/blob/master/notebooks/conditional_generation.ipynb) notebook for an example that uses encodings of Spotify track previews to influence the generation.
