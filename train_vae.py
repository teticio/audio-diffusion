# pip install -e git+https://github.com/CompVis/stable-diffusion.git@master
# pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
# convert_original_stable_diffusion_to_diffusers.py

# TODO
# grayscale
# log audio
# convert to huggingface / train huggingface

import os
import argparse

import torch
import torchvision
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from omegaconf import OmegaConf
from datasets import load_dataset
from librosa.util import normalize
from ldm.util import instantiate_from_config
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from audiodiffusion.mel import Mel


class AudioDiffusion(Dataset):

    def __init__(self, model_id):
        super().__init__()
        self.hf_dataset = load_dataset(model_id)['train']

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx]['image'].convert('RGB')
        image = np.frombuffer(image.tobytes(), dtype="uint8").reshape(
            (image.height, image.width, 3))
        image = ((image / 255) * 2 - 1)
        return {'image': image}


class AudioDiffusionDataModule(pl.LightningDataModule):

    def __init__(self, model_id, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = AudioDiffusion(model_id)
        self.num_workers = 1

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


# from https://github.com/CompVis/stable-diffusion/blob/main/main.py
class ImageLogger(Callback):

    def __init__(self,
                 batch_frequency,
                 max_images,
                 clamp=True,
                 increase_log_steps=True,
                 rescale=True,
                 disabled=False,
                 log_on_batch_idx=False,
                 log_first_step=False,
                 log_images_kwargs=None,
                 resolution=256,
                 hop_length=512):
        super().__init__()
        self.mel = Mel(x_res=resolution,
                       y_res=resolution,
                       hop_length=hop_length)
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._testtube,
        }
        self.log_steps = [
            2**n for n in range(int(np.log2(self.batch_freq)) + 1)
        ]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    #@rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            images_ = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = torchvision.utils.make_grid(images_)

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step)

            for _, image in enumerate(images_):
                image = (images_.numpy() *
                         255).round().astype("uint8").transpose(0, 2, 3, 1)
                audio = self.mel.image_to_audio(
                    Image.fromarray(image[0], mode='RGB').convert('L'))
                pl_module.logger.experiment.add_audio(
                    tag + f"/{_}",
                    normalize(audio),
                    global_step=pl_module.global_step,
                    sample_rate=self.mel.get_sample_rate())

    #@rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch,
                  batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx)
                and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch,
                                              split=split,
                                              **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            #self.log_local(pl_module.logger.save_dir, split, images,
            #               pl_module.global_step, pl_module.current_epoch,
            #               batch_idx)

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or
            (check_idx in self.log_steps)) and (check_idx > 0
                                                or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                #print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx):
        if not self.disabled and (pl_module.global_step > 0
                                  or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE using ldm.")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    config = OmegaConf.load('ldm_autoencoder_kl.yaml')
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(
        trainer_opt,
        callbacks=[
            ImageLogger(batch_frequency=1000,
                        max_images=8,
                        increase_log_steps=False,
                        log_on_batch_idx=True),
            ModelCheckpoint(dirpath='checkpoints',
                            filename='{epoch:06}',
                            verbose=True,
                            save_last=True)
        ])
    model = instantiate_from_config(config.model)
    model.learning_rate = config.model.base_learning_rate
    data = AudioDiffusionDataModule('teticio/audio-diffusion-256',
                                    batch_size=args.batch_size)
    trainer.fit(model, data)
