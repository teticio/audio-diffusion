from math import acos, sin
from typing import Iterable, Tuple, Union, List

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from librosa.beat import beat_track
from diffusers import (DiffusionPipeline, UNet2DConditionModel, DDIMScheduler,
                       DDPMScheduler, AutoencoderKL)

from .mel import Mel

VERSION = "1.2.6"


class AudioDiffusion:

    def __init__(self,
                 model_id: str = "teticio/audio-diffusion-256",
                 sample_rate: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 top_db: int = 80,
                 cuda: bool = torch.cuda.is_available(),
                 progress_bar: Iterable = tqdm):
        """Class for generating audio using De-noising Diffusion Probabilistic Models.

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

        sample_size = self.pipe.get_input_dims()
        self.mel = Mel(x_res=sample_size[1],
                       y_res=sample_size[0],
                       sample_rate=sample_rate,
                       n_fft=n_fft,
                       hop_length=hop_length,
                       top_db=top_db)

    def generate_spectrogram_and_audio(
        self,
        steps: int = None,
        generator: torch.Generator = None,
        step_generator: torch.Generator = None,
        eta: float = 0,
        noise: torch.Tensor = None
    ) -> Tuple[Image.Image, Tuple[int, np.ndarray]]:
        """Generate random mel spectrogram and convert to audio.

        Args:
            steps (int): number of de-noising steps (defaults to 50 for DDIM, 1000 for DDPM)
            generator (torch.Generator): random number generator or None
            step_generator (torch.Generator): random number generator used to de-noise or None
            eta (float): parameter between 0 and 1 used with DDIM scheduler
            noise (torch.Tensor): noisy image or None

        Returns:
            PIL Image: mel spectrogram
            (float, np.ndarray): sample rate and raw audio
        """
        images, (sample_rate,
                 audios) = self.pipe(mel=self.mel,
                                     batch_size=1,
                                     steps=steps,
                                     generator=generator,
                                     step_generator=step_generator,
                                     eta=eta,
                                     noise=noise)
        return images[0], (sample_rate, audios[0])

    def generate_spectrogram_and_audio_from_audio(
        self,
        audio_file: str = None,
        raw_audio: np.ndarray = None,
        slice: int = 0,
        start_step: int = 0,
        steps: int = None,
        generator: torch.Generator = None,
        mask_start_secs: float = 0,
        mask_end_secs: float = 0,
        step_generator: torch.Generator = None,
        eta: float = 0,
        noise: torch.Tensor = None
    ) -> Tuple[Image.Image, Tuple[int, np.ndarray]]:
        """Generate random mel spectrogram from audio input and convert to audio.

        Args:
            audio_file (str): must be a file on disk due to Librosa limitation or
            raw_audio (np.ndarray): audio as numpy array
            slice (int): slice number of audio to convert
            start_step (int): step to start from
            steps (int): number of de-noising steps (defaults to 50 for DDIM, 1000 for DDPM)
            generator (torch.Generator): random number generator or None
            mask_start_secs (float): number of seconds of audio to mask (not generate) at start
            mask_end_secs (float): number of seconds of audio to mask (not generate) at end
            step_generator (torch.Generator): random number generator used to de-noise or None
            eta (float): parameter between 0 and 1 used with DDIM scheduler
            noise (torch.Tensor): noisy image or None

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
                                     mask_end_secs=mask_end_secs,
                                     step_generator=step_generator,
                                     eta=eta,
                                     noise=noise)
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

    def get_input_dims(self) -> Tuple:
        """Returns dimension of input image

        Returns:
            Tuple: (height, width)
        """
        input_module = self.vqvae if hasattr(self, 'vqvae') else self.unet
        # For backwards compatibility
        sample_size = (
            input_module.sample_size, input_module.sample_size) if type(
                input_module.sample_size) == int else input_module.sample_size
        return sample_size

    def get_default_steps(self) -> int:
        """Returns default number of steps recommended for inference

        Returns:
            int: number of steps
        """
        return 50 if isinstance(self.scheduler, DDIMScheduler) else 1000

    @torch.no_grad()
    def __call__(
        self,
        mel: Mel,
        batch_size: int = 1,
        audio_file: str = None,
        raw_audio: np.ndarray = None,
        slice: int = 0,
        start_step: int = 0,
        steps: int = None,
        generator: torch.Generator = None,
        mask_start_secs: float = 0,
        mask_end_secs: float = 0,
        step_generator: torch.Generator = None,
        eta: float = 0,
        noise: torch.Tensor = None
    ) -> Tuple[List[Image.Image], Tuple[int, List[np.ndarray]]]:
        """Generate random mel spectrogram from audio input and convert to audio.

        Args:
            mel (Mel): instance of Mel class to perform image <-> audio
            batch_size (int): number of samples to generate
            audio_file (str): must be a file on disk due to Librosa limitation or
            raw_audio (np.ndarray): audio as numpy array
            slice (int): slice number of audio to convert
            start_step (int): step to start from
            steps (int): number of de-noising steps (defaults to 50 for DDIM, 1000 for DDPM)
            generator (torch.Generator): random number generator or None
            mask_start_secs (float): number of seconds of audio to mask (not generate) at start
            mask_end_secs (float): number of seconds of audio to mask (not generate) at end
            step_generator (torch.Generator): random number generator used to de-noise or None
            eta (float): parameter between 0 and 1 used with DDIM scheduler
            noise (torch.Tensor): noise tensor of shape (batch_size, 1, height, width) or None

        Returns:
            List[PIL Image]: mel spectrograms
            (float, List[np.ndarray]): sample rate and raw audios
        """

        steps = steps or self.get_default_steps()
        self.scheduler.set_timesteps(steps)
        step_generator = step_generator or generator
        # For backwards compatibility
        if type(self.unet.sample_size) == int:
            self.unet.sample_size = (self.unet.sample_size,
                                     self.unet.sample_size)
        if noise is None:
            noise = torch.randn(
                (batch_size, self.unet.in_channels, self.unet.sample_size[0],
                 self.unet.sample_size[1]),
                generator=generator)
        images = noise
        mask = None

        if audio_file is not None or raw_audio is not None:
            mel.load_audio(audio_file, raw_audio)
            input_image = mel.audio_slice_to_image(slice)
            input_image = np.frombuffer(input_image.tobytes(),
                                        dtype="uint8").reshape(
                                            (input_image.height,
                                             input_image.width))
            input_image = ((input_image / 255) * 2 - 1)
            input_images = torch.tensor(input_image[np.newaxis, :, :],
                                        dtype=torch.float)

            if hasattr(self, 'vqvae'):
                input_images = self.vqvae.encode(
                    torch.unsqueeze(input_images,
                                    0).to(self.device)).latent_dist.sample(
                                        generator=generator).cpu()[0]
                input_images = 0.18215 * input_images

            if start_step > 0:
                images[0, 0] = self.scheduler.add_noise(
                    input_images, noise,
                    self.scheduler.timesteps[start_step - 1])

            pixels_per_second = (self.unet.sample_size[1] *
                                 mel.get_sample_rate() / mel.x_res /
                                 mel.hop_length)
            mask_start = int(mask_start_secs * pixels_per_second)
            mask_end = int(mask_end_secs * pixels_per_second)
            mask = self.scheduler.add_noise(
                input_images, noise,
                torch.tensor(self.scheduler.timesteps[start_step:]))

        images = images.to(self.device)
        for step, t in enumerate(
                self.progress_bar(self.scheduler.timesteps[start_step:])):
            model_output = self.unet(images, t)['sample']

            if isinstance(self.scheduler, DDIMScheduler):
                images = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=images,
                    eta=eta,
                    generator=step_generator)['prev_sample']
            else:
                images = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=images,
                    generator=step_generator)['prev_sample']

            if mask is not None:
                if mask_start > 0:
                    images[:, :, :, :mask_start] = mask[:,
                                                        step, :, :mask_start]
                if mask_end > 0:
                    images[:, :, :, -mask_end:] = mask[:, step, :, -mask_end:]

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

    @torch.no_grad()
    def encode(self, images: List[Image.Image], steps: int = 50) -> np.ndarray:
        """Reverse step process: recover noisy image from generated image.

        Args:
            images (List[PIL Image]): list of images to encode
            steps (int): number of encoding steps to perform (defaults to 50)

        Returns:
            np.ndarray: noise tensor of shape (batch_size, 1, height, width)
        """

        # Only works with DDIM as this method is deterministic
        assert isinstance(self.scheduler, DDIMScheduler)
        self.scheduler.set_timesteps(steps)
        sample = np.array([
            np.frombuffer(image.tobytes(), dtype="uint8").reshape(
                (1, image.height, image.width)) for image in images
        ])
        sample = ((sample / 255) * 2 - 1)
        sample = torch.Tensor(sample).to(self.device)

        for t in self.progress_bar(torch.flip(self.scheduler.timesteps,
                                              (0, ))):
            prev_timestep = (t - self.scheduler.num_train_timesteps //
                             self.scheduler.num_inference_steps)
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (self.scheduler.alphas_cumprod[prev_timestep]
                                 if prev_timestep >= 0 else
                                 self.scheduler.final_alpha_cumprod)
            beta_prod_t = 1 - alpha_prod_t
            model_output = self.unet(sample, t)['sample']
            pred_sample_direction = (1 -
                                     alpha_prod_t_prev)**(0.5) * model_output
            sample = (sample -
                      pred_sample_direction) * alpha_prod_t_prev**(-0.5)
            sample = sample * alpha_prod_t**(0.5) + beta_prod_t**(
                0.5) * model_output

        return sample

    @staticmethod
    def slerp(x0: torch.Tensor, x1: torch.Tensor,
              alpha: float) -> torch.Tensor:
        """Spherical Linear intERPolation

        Args:
            x0 (torch.Tensor): first tensor to interpolate between
            x1 (torch.Tensor): seconds tensor to interpolate between
            alpha (float): interpolation between 0 and 1

        Returns:
            torch.Tensor: interpolated tensor
        """

        theta = acos(
            torch.dot(torch.flatten(x0), torch.flatten(x1)) / torch.norm(x0) /
            torch.norm(x1))
        return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(
            alpha * theta) * x1 / sin(theta)


class LatentAudioDiffusionPipeline(AudioDiffusionPipeline):

    def __init__(self, unet: UNet2DConditionModel,
                 scheduler: Union[DDIMScheduler,
                                  DDPMScheduler], vqvae: AutoencoderKL):
        super().__init__(unet=unet, scheduler=scheduler)
        self.register_modules(vqvae=vqvae)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
