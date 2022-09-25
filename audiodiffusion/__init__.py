from typing import Iterable, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from diffusers import DDPMPipeline
from librosa.beat import beat_track

from .mel import Mel

VERSION = "1.1.2"


class AudioDiffusion:

    def __init__(self,
                 model_id: str = "teticio/audio-diffusion-256",
                 resolution: int = 256,
                 cuda: bool = torch.cuda.is_available(),
                 progress_bar: Iterable = tqdm):
        """Class for generating audio using Denoising Diffusion Probabilistic Models.

        Args:
            model_id (String): name of model (local directory or Hugging Face Hub)
            resolution (int): size of square mel spectrogram in pixels
            cuda (bool): use CUDA?
            progress_bar (iterable): iterable callback for progress updates or None
        """
        self.mel = Mel(x_res=resolution, y_res=resolution)
        self.model_id = model_id
        self.ddpm = DDPMPipeline.from_pretrained(self.model_id)
        if cuda:
            self.ddpm.to("cuda")
        self.progress_bar = progress_bar or (lambda _: _)

    def generate_spectrogram_and_audio(
        self,
        generator: torch.Generator = None
    ) -> Tuple[Image.Image, Tuple[int, np.ndarray]]:
        """Generate random mel spectrogram and convert to audio.

        Args:
            generator (torch.Generator): random number generator or None

        Returns:
            PIL Image: mel spectrogram
            (float, np.ndarray): sample rate and raw audio
        """
        images = self.ddpm(output_type="numpy", generator=generator)["sample"]
        images = (images * 255).round().astype("uint8").transpose(0, 3, 1, 2)
        image = Image.fromarray(images[0][0])
        audio = self.mel.image_to_audio(image)
        return image, (self.mel.get_sample_rate(), audio)

    @torch.no_grad()
    def generate_spectrogram_and_audio_from_audio(
        self,
        audio_file: str = None,
        raw_audio: np.ndarray = None,
        slice: int = 0,
        start_step: int = 0,
        steps: int = 1000,
        generator: torch.Generator = None
    ) -> Tuple[Image.Image, Tuple[int, np.ndarray]]:
        """Generate random mel spectrogram from audio input and convert to audio.

        Args:
            audio_file (str): must be a file on disk due to Librosa limitation or
            raw_audio (np.ndarray): audio as numpy array
            slice (int): slice number of audio to convert
            start_step (int): step to start from
            steps (int): number of de-noising steps to perform
            generator (torch.Generator): random number generator or None

        Returns:
            PIL Image: mel spectrogram
            (float, np.ndarray): sample rate and raw audio
        """

        # It would be better to derive a class from DDPMDiffusionPipeline
        # but currently the return type ImagePipelineOutput cannot be imported.
        images = torch.randn(
            (1, self.ddpm.unet.in_channels, self.ddpm.unet.sample_size,
             self.ddpm.unet.sample_size),
            generator=generator,
        )
        if audio_file is not None or raw_audio is not None:
            self.mel.load_audio(audio_file, raw_audio)
            input_image = self.mel.audio_slice_to_image(slice)
            input_image = np.frombuffer(input_image.tobytes(),
                                        dtype="uint8").reshape(
                                            (input_image.width,
                                             input_image.height))
            input_image = ((input_image / 255) * 2 - 1)
            if start_step > 0:
                images[0][0] = self.ddpm.scheduler.add_noise(
                torch.tensor(input_image[np.newaxis, np.newaxis, :]), images,
                steps - start_step)

        images = images.to(self.ddpm.device)
        self.ddpm.scheduler.set_timesteps(steps)
        for t in self.progress_bar(self.ddpm.scheduler.timesteps[start_step:]):
            model_output = self.ddpm.unet(images, t)['sample']
            images = self.ddpm.scheduler.step(
                model_output, t, images, generator=generator)['prev_sample']
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()

        images = (images * 255).round().astype("uint8").transpose(0, 3, 1, 2)
        image = Image.fromarray(images[0][0])
        audio = self.mel.image_to_audio(image)
        return image, (self.mel.get_sample_rate(), audio)

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
