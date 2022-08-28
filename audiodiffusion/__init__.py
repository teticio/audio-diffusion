from PIL import Image
from torch import cuda
from diffusers import DDPMPipeline

from .mel import Mel

VERSION = "1.0.1"


class AudioDiffusion:

    def __init__(self,
                 model_id="teticio/audio-diffusion-256",
                 resolution=256,
                 cuda=cuda.is_available()):
        """Class for generating audio using Denoising Diffusion Probabilistic Models.

        Args:
            model_id (String): name of model (local directory or Hugging Face Hub)
            resolution (int): size of square mel spectrogram in pixels
            cuda (bool): use CUDA?
        """
        self.mel = Mel(x_res=resolution, y_res=resolution)
        self.model_id = model_id
        self.ddpm = DDPMPipeline.from_pretrained(self.model_id)
        if cuda:
            self.ddpm.to("cuda")

    def generate_spectrogram_and_audio(self):
        """Generate random mel spectrogram and convert to audio.

        Returns:
            PIL Image: mel spectrogram
            (float, array): sample rate and raw audio
        """
        images = self.ddpm(output_type="numpy")["sample"]
        images = (images * 255).round().astype("uint8").transpose(0, 3, 1, 2)
        image = Image.fromarray(images[0][0])
        audio = self.mel.image_to_audio(image)
        return image, (self.mel.get_sample_rate(), audio)
