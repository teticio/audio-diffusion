# audio-diffusion
```bash
python src/audio_to_images.py \
  --resolution=256 \
  --input_dir=path-to-audio-files \
  --output_dir=data
```
```bash
accelerate launch src/train_unconditional.py \
  --dataset_name="data" \
  --resolution=256 \
  --output_dir="ddpm-ema-audio-256" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no
```
