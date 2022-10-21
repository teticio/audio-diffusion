from typing import Iterable, Tuple, Union, List

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from librosa.beat import beat_track
from diffusers import (DiffusionPipeline, DDPMPipeline, UNet2DConditionModel,
                       DDIMScheduler, DDPMScheduler, AutoencoderKL)

from .mel import Mel

VERSION = "1.2.1"


class AudioDiffusion:

    def __init__(self,
                 model_id: str = "teticio/audio-diffusion-256",
                 sample_rate: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 top_db: int = 80,
                 cuda: bool = torch.cuda.is_available(),
                 progress_bar: Iterable = tqdm):
        """Class for generating audio using Denoising Diffusion Probabilistic Models.

        Args:
            model_id (String): name of model (local directory or Hugging Face Hub)
            sample_rate (int): sample rate of audio
            n_fft (int): number of Fast Fourier Transforms
            hop_length (int): hop length (a higher number is recommended for lower than 256 y_res)
            top_db (int): loudest in decibels
            cuda (bool): use CUDA?
            progress_bar (iterable): iterable callback for progress updates or None
        """
        self.model_id = model_id
        pipeline = {
            'LatentAudioDiffusionPipeline': LatentAudioDiffusionPipeline,
            'AudioDiffusionPipeline': AudioDiffusionPipeline
        }.get(
            DiffusionPipeline.get_config_dict(self.model_id)['_class_name'],
            AudioDiffusionPipeline)
        self.pipe = pipeline.from_pretrained(self.model_id)
        if cuda:
            self.pipe.to("cuda")
        self.progress_bar = progress_bar or (lambda _: _)

        # For backwards compatibility
        sample_size = (self.pipe.unet.sample_size,
                       self.pipe.unet.sample_size) if type(
                           self.pipe.unet.sample_size
                       ) == int else self.pipe.unet.sample_size
        self.mel = Mel(x_res=sample_size[1],
                       y_res=sample_size[0],
                       sample_rate=sample_rate,
                       n_fft=n_fft,
                       hop_length=hop_length,
                       top_db=top_db)

    def generate_spectrogram_and_audio(
        self,
        steps: int = 1000,
        generator: torch.Generator = None
    ) -> Tuple[Image.Image, Tuple[int, np.ndarray]]:
        """Generate random mel spectrogram and convert to audio.

        Args:
            steps (int): number of de-noising steps to perform (defaults to num_train_timesteps)
            generator (torch.Generator): random number generator or None

        Returns:
            PIL Image: mel spectrogram
            (float, np.ndarray): sample rate and raw audio
        """
        images, (sample_rate, audios) = self.pipe(mel=self.mel,
                                                  batch_size=1,
                                                  steps=steps,
                                                  generator=generator)
        return images[0], (sample_rate, audios[0])

    def generate_spectrogram_and_audio_from_audio(
        self,
        audio_file: str = None,
        raw_audio: np.ndarray = None,
        slice: int = 0,
        start_step: int = 0,
        steps: int = 1000,
        generator: torch.Generator = None,
        mask_start_secs: float = 0,
        mask_end_secs: float = 0
    ) -> Tuple[Image.Image, Tuple[int, np.ndarray]]:
        """Generate random mel spectrogram from audio input and convert to audio.

        Args:
            audio_file (str): must be a file on disk due to Librosa limitation or
            raw_audio (np.ndarray): audio as numpy array
            slice (int): slice number of audio to convert
            start_step (int): step to start from
            steps (int): number of de-noising steps to perform (defaults to num_train_timesteps)
            generator (torch.Generator): random number generator or None
            mask_start_secs (float): number of seconds of audio to mask (not generate) at start
            mask_end_secs (float): number of seconds of audio to mask (not generate) at end

        Returns:
            PIL Image: mel spectrogram
            (float, np.ndarray): sample rate and raw audio
        """

        images, (sample_rate,
                 audios) = self.pipe(mel=self.mel,
                                     batch_size=1,
                                     audio_file=audio_file,
                                     raw_audio=raw_audio,
                                     slice=slice,
                                     start_step=start_step,
                                     steps=steps,
                                     generator=generator,
                                     mask_start_secs=mask_start_secs,
                                     mask_end_secs=mask_end_secs)
        return images[0], (sample_rate, audios[0])

    @staticmethod
    def loop_it(audio: np.ndarray,
                sample_rate: int,
                loops: int = 12) -> np.ndarray:
        """Loop audio

        Args:
            audio (np.ndarray): audio as numpy array
            sample_rate (int): sample rate of audio
            loops (int): number of times to loop

        Returns:
            (float, np.ndarray): sample rate and raw audio or None
        """
        _, beats = beat_track(y=audio, sr=sample_rate, units='samples')
        for beats_in_bar in [16, 12, 8, 4]:
            if len(beats) > beats_in_bar:
                return np.tile(audio[beats[0]:beats[beats_in_bar]], loops)
        return None


class AudioDiffusionPipeline(DiffusionPipeline):

    def __init__(self, unet: UNet2DConditionModel,
                 scheduler: Union[DDIMScheduler, DDPMScheduler]):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        mel: Mel,
        batch_size: int = 1,
        audio_file: str = None,
        raw_audio: np.ndarray = None,
        slice: int = 0,
        start_step: int = 0,
        steps: int = 1000,
        generator: torch.Generator = None,
        mask_start_secs: float = 0,
        mask_end_secs: float = 0
    ) -> Tuple[List[Image.Image], Tuple[int, List[np.ndarray]]]:
        """Generate random mel spectrogram from audio input and convert to audio.

        Args:
            mel (Mel): instance of Mel class to perform image <-> audio
            batch_size (int): number of samples to generate
            audio_file (str): must be a file on disk due to Librosa limitation or
            raw_audio (np.ndarray): audio as numpy array
            slice (int): slice number of audio to convert
            start_step (int): step to start from
            steps (int): number of de-noising steps to perform (defaults to num_train_timesteps)
            generator (torch.Generator): random number generator or None
            mask_start_secs (float): number of seconds of audio to mask (not generate) at start
            mask_end_secs (float): number of seconds of audio to mask (not generate) at end

        Returns:
            List[PIL Image]: mel spectrograms
            (float, List[np.ndarray]): sample rate and raw audios
        """

        self.scheduler.set_timesteps(steps)
        mask = None
        images = noise = torch.randn(
            (batch_size, self.unet.in_channels, mel.y_res, mel.x_res),
            generator=generator)

        if audio_file is not None or raw_audio is not None:
            mel.load_audio(audio_file, raw_audio)
            input_image = mel.audio_slice_to_image(slice)
            input_image = np.frombuffer(input_image.tobytes(),
                                        dtype="uint8").reshape(
                                            (input_image.height,
                                             input_image.width))
            input_image = ((input_image / 255) * 2 - 1)
            input_images = np.tile(input_image, (batch_size, 1, 1, 1))

            if hasattr(self, 'vqvae'):
                input_images = self.vqvae.encode(
                    input_images).latent_dist.sample(generator=generator)
                input_images = 0.18215 * input_images

            if start_step > 0:
                images[0, 0] = self.scheduler.add_noise(
                    torch.tensor(input_images[:, np.newaxis, np.newaxis, :]),
                    noise, torch.tensor(steps - start_step))

            pixels_per_second = (mel.get_sample_rate() / mel.hop_length)
            mask_start = int(mask_start_secs * pixels_per_second)
            mask_end = int(mask_end_secs * pixels_per_second)
            mask = self.scheduler.add_noise(
                torch.tensor(input_images[:, np.newaxis, :]), noise,
                torch.tensor(self.scheduler.timesteps[start_step:]))

        images = images.to(self.device)
        for step, t in enumerate(
                self.progress_bar(self.scheduler.timesteps[start_step:])):
            model_output = self.unet(images, t)['sample']
            images = self.scheduler.step(model_output,
                                         t,
                                         images,
                                         generator=generator)['prev_sample']

            if mask is not None:
                if mask_start > 0:
                    images[:, :, :, :mask_start] = mask[
                        step, :, :, :, :mask_start]
                if mask_end > 0:
                    images[:, :, :, -mask_end:] = mask[step, :, :, :,
                                                       -mask_end:]

        if hasattr(self, 'vqvae'):
            # 0.18215 was scaling factor used in training to ensure unit variance
            images = 1 / 0.18215 * images
            images = self.vqvae.decode(images)['sample']

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = list(
            map(lambda _: Image.fromarray(_[:, :, 0]), images) if images.
            shape[3] == 1 else map(
                lambda _: Image.fromarray(_, mode='RGB').convert('L'), images))

        audios = list(map(lambda _: mel.image_to_audio(_), images))
        return images, (mel.get_sample_rate(), audios)


class LatentAudioDiffusionPipeline(AudioDiffusionPipeline):

    def __init__(self, unet: UNet2DConditionModel,
                 scheduler: Union[DDIMScheduler,
                                  DDPMScheduler], vqvae: AutoencoderKL):
        super().__init__(unet=unet, scheduler=scheduler)
        self.register_modules(vqvae=vqvae)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
