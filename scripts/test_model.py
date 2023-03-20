import torch
import random
import librosa
import numpy as np
from datasets import load_dataset
from audiodiffusion import AudioDiffusion
import torchaudio
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile



def main():

    if not os.path.exists("images"):
        os.makedirs("images")
    if not os.path.exists("audio"):
        os.makedirs("audio")

    device = "cuda" if torch.cuda.is_available() else "mps"
    generator = torch.Generator(device=device)

    model_id = "models/Atlantic_Spotted_Dolphin_128_ddim"
    audio_diffusion = AudioDiffusion(model_id=model_id)
    audio_diffusion.pipe.to("mps")
    mel = audio_diffusion.pipe.mel

    for i in range(1):
        seed = generator.seed()
        print(f'Seed = {seed}')
        generator.manual_seed(seed)
        image, (sample_rate,
                audio) = audio_diffusion.generate_spectrogram_and_audio(
                    generator=generator)
        image_file = f"images/spectrogram_{i}.png"
        image.save(image_file)
        print(np.array(image))
        audio_file = f"audio/audio_{i}.wav"
        wavfile.write(audio_file, sample_rate, audio)
    
if __name__ == "__main__":
    main()

