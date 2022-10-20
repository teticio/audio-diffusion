# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py

import argparse
import os

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_from_disk, load_dataset
from diffusers import (DiffusionPipeline, DDPMScheduler, UNet2DModel,
                       DDIMScheduler, AutoencoderKL)
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
import numpy as np
from tqdm.auto import tqdm
from librosa.util import normalize

from audiodiffusion.mel import Mel
from audiodiffusion import LatentAudioDiffusionPipeline, AudioDiffusionPipeline

logger = get_logger(__name__)


def main(args):
    output_dir = os.environ.get("SM_MODEL_DIR", None) or args.output_dir
    logging_dir = os.path.join(output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    if args.dataset_name is not None:
        if os.path.exists(args.dataset_name):
            dataset = load_from_disk(args.dataset_name,
                                     args.dataset_config_name)["train"]
        else:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                use_auth_token=True if args.use_auth_token else None,
                split="train",
            )
    else:
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
            split="train",
        )
    # Determine image resolution
    resolution = dataset[0]['image'].height, dataset[0]['image'].width

    augmentations = Compose([
        ToTensor(),
        Normalize([0.5], [0.5]),
    ])

    def transforms(examples):
        if args.vae is not None and vqvae.config['in_channels'] == 3:
            images = [
                augmentations(image.convert('RGB'))
                for image in examples["image"]
            ]
        else:
            images = [augmentations(image) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True)

    vqvae = None
    if args.vae is not None:
        vqvae = AutoencoderKL.from_pretrained(args.vae)
        # Determine latent resolution
        with torch.no_grad():
            latent_resolution = vqvae.encode(
                torch.zeros((1, 1) +
                            resolution)).latent_dist.sample().shape[2:]

    if args.from_pretrained is not None:
        pipeline = DiffusionPipeline.from_pretrained(args.from_pretrained)
        model = pipeline.unet
        if hasattr(pipeline, 'vqvae'):
            vqvae = AutoencoderKL.from_pretrained(args.vae)
    else:
        model = UNet2DModel(
            sample_size=resolution if vqvae is None else latent_resolution,
            in_channels=1
            if vqvae is None else vqvae.config['latent_channels'],
            out_channels=1
            if vqvae is None else vqvae.config['latent_channels'],
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_steps, tensor_format="pt")
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_steps, tensor_format="pt")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) //
        args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    ema_model = EMAModel(
        getattr(model, "module", model),
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        max_value=args.ema_max_decay,
    )

    if args.push_to_hub:
        repo = init_git_repo(args, at_init=True)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    mel = Mel(x_res=resolution[1],
              y_res=resolution[0],
              hop_length=args.hop_length)

    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
            if epoch == args.start_epoch - 1 and args.use_ema:
                ema_model.optimization_step = global_step
            continue

        model.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]

            if vqvae is not None:
                vqvae.to(clean_images.device)
                with torch.no_grad():
                    clean_images = vqvae.encode(
                        clean_images).latent_dist.sample()
                # Scale latent images to ensure approximately unit variance
                clean_images = clean_images * 0.18215

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bsz, ),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                if vqvae is not None:
                    pipeline = LatentAudioDiffusionPipeline(
                        unet=accelerator.unwrap_model(
                            ema_model.averaged_model if args.use_ema else model
                        ),
                        vqvae=vqvae,
                        scheduler=noise_scheduler)
                else:
                    pipeline = AudioDiffusionPipeline(
                        unet=accelerator.unwrap_model(
                            ema_model.averaged_model if args.use_ema else model
                        ),
                        scheduler=noise_scheduler,
                    )

                # save the model
                if args.push_to_hub:
                    try:
                        push_to_hub(
                            args,
                            pipeline,
                            repo,
                            commit_message=f"Epoch {epoch}",
                            blocking=False,
                        )
                    except NameError:  # current version of diffusers has a little bug
                        pass
                else:
                    pipeline.save_pretrained(output_dir)

            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                generator = torch.manual_seed(42)
                # run pipeline in inference (sample random noise and denoise)
                images, (sample_rate, audios) = pipeline(
                    mel=mel,
                    generator=generator,
                    batch_size=args.eval_batch_size,
                )

                # denormalize the images and save to tensorboard
                images = np.array([
                    np.frombuffer(image.tobytes(), dtype="uint8").reshape(
                        (len(image.getbands()), image.height, image.width))
                    for image in images
                ])
                accelerator.trackers[0].writer.add_images(
                    "test_samples", images, epoch)
                for _, audio in enumerate(audios):
                    accelerator.trackers[0].writer.add_audio(
                        f"test_audio_{_}",
                        normalize(audio),
                        epoch,
                        sample_rate=sample_rate,
                    )
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument("--output_dir", type=str, default="ddpm-model-64")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--use_auth_token", type=bool, default=False)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."),
    )
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--scheduler",
                        type=str,
                        default="ddpm",
                        help="ddpm or ddim")
    parser.add_argument("--vae",
                        type=str,
                        default=None,
                        help="pretrained VAE model for latent diffusion")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )
    if args.dataset_name is not None and args.dataset_name == args.hub_model_id:
        raise ValueError(
            "The local dataset name must be different from the hub model id.")

    main(args)
