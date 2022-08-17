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

# audio-diffusion

### Apply [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) using the new Hugging Face [diffusers](https://github.com/huggingface/diffusers) package to synthesize music instead of images.

---

![mel spectrogram](mel.png)

Audio can be represented as images by transforming to a [mel spectrogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), such as the one shown above. The class `Mel` in `mel.py` can convert a slice of audio into a mel spectrogram of `x_res` x `y_res` and vice versa. The higher the resolution, the less audio information will be lost. You can see how this works in the `test-mel.ipynb` notebook.

A DDPM model is trained on a set of mel spectrograms that have been generated from a directory of audio files. It is then used to synthesize similar mel spectrograms, which are then converted back into audio. See the `test-model.ipynb` notebook for an example.

You can play around with the model I trained on about 500 songs from my Spotify "liked" playlist on [Google Colab](https://colab.research.google.com/github/teticio/audio-diffusion/blob/master/notebooks/test-model.ipynb) or [Hugging Face spaces](https://huggingface.co/spaces/teticio/audio-diffusion).

## Generate Mel spectrogram dataset from directory of audio files
#### Training can be run with Mel spectrograms of resolution 64x64 on a single commercial grade GPU (e.g. RTX 2080 Ti). The `hop_length` should be set to 1024 for better results.

```bash
python src/audio_to_images.py \
  --resolution 64 \
  --hop_length 1024\
  --input_dir path-to-audio-files \
  --output_dir data-test
```

#### Generate dataset of 256x256 Mel spectrograms and push to hub (you will need to be authenticated with `huggingface-cli login`).

```bash
python src/audio_to_images.py \
  --resolution 256 \
  --input_dir path-to-audio-files \
  --output_dir data-256 \
  --push_to_hub teticio\audio-diffusion-256
```
## Train model
#### Run training on local machine.

```bash
accelerate launch --config_file accelerate_local.yaml \
  src/train_unconditional.py \
  --dataset_name data-64 \
  --resolution 64 \
  --hop_length 1024 \
  --output_dir ddpm-ema-audio-64 \
  --train_batch_size 16 \
  --num_epochs 100 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_warmup_steps 500 \
  --mixed_precision no
```

#### Run training on local machine with `batch_size` of 2 and `gradient_accumulation_steps` 8 to compensate, so that 256x256 resolution model fits on commercial grade GPU and push to hub.

```bash
accelerate launch --config_file accelerate_local.yaml \
  src/train_unconditional.py \
  --dataset_name teticio/audio-diffusion-256 \
  --resolution 256 \
  --output_dir ddpm-ema-audio-256 \
  --num_epochs 100 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_warmup_steps 500 \
  --mixed_precision no \
  --push_to_hub True \
  --hub_model_id teticio/audio-diffusion-256 \
  --hub_token $(cat $HOME/.huggingface/token)
```

#### Run training on SageMaker.

```bash
accelerate launch --config_file accelerate_sagemaker.yaml \
  src/train_unconditional.py \
  --dataset_name teticio/audio-diffusion-256 \
  --resolution 256 \
  --output_dir ddpm-ema-audio-256 \
  --train_batch_size 16 \
  --num_epochs 100 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_warmup_steps 500 \
  --mixed_precision no
```
