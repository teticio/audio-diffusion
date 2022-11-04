import os
import argparse

import torch
import torchvision
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from omegaconf import OmegaConf
from librosa.util import normalize
from ldm.util import instantiate_from_config
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk, load_dataset
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

from audiodiffusion.mel import Mel
from audiodiffusion.utils import convert_ldm_to_hf_vae


class AudioDiffusion(Dataset):

    def __init__(self, model_id, channels=3):
        super().__init__()
        self.channels = channels
        if os.path.exists(model_id):
            self.hf_dataset = load_from_disk(model_id)['train']
        else:
            self.hf_dataset = load_dataset(model_id)['train']

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx]['image']
        if self.channels == 3:
            image = image.convert('RGB')
        image = np.frombuffer(image.tobytes(), dtype="uint8").reshape(
            (image.height, image.width, self.channels))
        image = ((image / 255) * 2 - 1)
        return {'image': image}


class AudioDiffusionDataModule(pl.LightningDataModule):

    def __init__(self, model_id, batch_size, channels):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = AudioDiffusion(model_id=model_id, channels=channels)
        self.num_workers = 1

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


class ImageLogger(Callback):

    def __init__(self, every=1000, hop_length=512):
        super().__init__()
        self.every = every
        self.hop_length = hop_length

    @rank_zero_only
    def log_images_and_audios(self, pl_module, batch):
        pl_module.eval()
        with torch.no_grad():
            images = pl_module.log_images(batch, split='train')
        pl_module.train()

        image_shape = next(iter(images.values())).shape
        channels = image_shape[1]
        mel = Mel(x_res=image_shape[2],
                  y_res=image_shape[3],
                  hop_length=self.hop_length)

        for k in images:
            images[k] = images[k].detach().cpu()
            images[k] = torch.clamp(images[k], -1., 1.)
            images[k] = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = torchvision.utils.make_grid(images[k])

            tag = f"train/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step)

            images[k] = (images[k].numpy() *
                         255).round().astype("uint8").transpose(0, 2, 3, 1)
            for _, image in enumerate(images[k]):
                audio = mel.image_to_audio(
                    Image.fromarray(image, mode='RGB').convert('L')
                    if channels == 3 else Image.fromarray(image[:, :, 0]))
                pl_module.logger.experiment.add_audio(
                    tag + f"/{_}",
                    normalize(audio),
                    global_step=pl_module.global_step,
                    sample_rate=mel.get_sample_rate())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx):
        if (batch_idx + 1) % self.every != 0:
            return
        self.log_images_and_audios(pl_module, batch)


class HFModelCheckpoint(ModelCheckpoint):

    def __init__(self, ldm_config, hf_checkpoint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ldm_config = ldm_config
        self.hf_checkpoint = hf_checkpoint

    def on_train_epoch_end(self, trainer, pl_module):
        ldm_checkpoint = self._get_metric_interpolated_filepath_name(
            {'epoch': trainer.current_epoch}, trainer)
        super().on_train_epoch_end(trainer, pl_module)
        convert_ldm_to_hf_vae(ldm_checkpoint, self.ldm_config,
                              self.hf_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE using ldm.")
    parser.add_argument("-d", "--dataset_name", type=str, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-c",
                        "--ldm_config_file",
                        type=str,
                        default="config/ldm_autoencoder_kl.yaml")
    parser.add_argument("--ldm_checkpoint_dir",
                        type=str,
                        default="models/ldm-autoencoder-kl")
    parser.add_argument("--hf_checkpoint_dir",
                        type=str,
                        default="models/autoencoder-kl")
    parser.add_argument("-r",
                        "--resume_from_checkpoint",
                        type=str,
                        default=None)
    parser.add_argument("-g",
                        "--gradient_accumulation_steps",
                        type=int,
                        default=1)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--save_images_batches", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()

    config = OmegaConf.load(args.ldm_config_file)
    model = instantiate_from_config(config.model)
    model.learning_rate = config.model.base_learning_rate
    data = AudioDiffusionDataModule(
        model_id=args.dataset_name,
        batch_size=args.batch_size,
        channels=config.model.params.ddconfig.in_channels)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config.accumulate_grad_batches = args.gradient_accumulation_steps
    trainer_opt = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(
        trainer_opt,
        max_epochs=args.max_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
        callbacks=[
            ImageLogger(every=args.save_images_batches,
                        hop_length=args.hop_length),
            HFModelCheckpoint(ldm_config=config,
                              hf_checkpoint=args.hf_checkpoint_dir,
                              dirpath=args.ldm_checkpoint_dir,
                              filename='{epoch:06}',
                              verbose=True,
                              save_last=True)
        ])
    trainer.fit(model, data)
