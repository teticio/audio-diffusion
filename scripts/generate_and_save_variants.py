import re
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
import pandas as pd
from audiodiffusion.pipeline_audio_diffusion import AudioDiffusionPipeline

def get_train_audio_files(metadata_path: str, base_path: str = ""):
    df = pd.read_csv(metadata_path)

    paths = []
    for _, row in df.iterrows():
        paths.append(row['path'].split("data/")[-1])
    return paths


def generate_variants(model_id, start_step, input_dir, metadata_path = None, task_name: str = ""):

    if not os.path.exists("images"):
        os.makedirs("images")
    if not os.path.exists("audio"):
        os.makedirs("audio")
    if not os.path.exists(f"augmentations/{task_name}"):
        os.makedirs(f"augmentations/{task_name}")

    device = "cuda" if torch.cuda.is_available() else "mps"
    generator = torch.Generator(device=device)



    audio_diffusion = AudioDiffusion(model_id=model_id)
    audio_diffusion.pipe.to("mps")
    mel = audio_diffusion.pipe.mel


    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files
        if re.search("\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    print("starting audios", len(audio_files))
    if metadata_path:
        train_files = get_train_audio_files(metadata_path=metadata_path)
        audio_files = [file for file in audio_files if file.split("data/")[-1] in train_files]
        print("modified audios", len(audio_files))

    for i in range(1):
        for audio_file_path in audio_files:
            if "common" in audio_file_path:
                continue
            # sr, audio_file = wavfile.read(audio_file_path)
            audio_file, sr = librosa.load(audio_file_path)
            print(f"sr {sr} and audio {audio_file}")

            seed = generator.seed()
            print(f'Seed = {seed}')
            generator.manual_seed(seed)
            image, (sample_rate,
                    audio) = audio_diffusion.generate_spectrogram_and_audio_from_audio(
                        raw_audio=audio_file, generator=generator, start_step = start_step)
            audio_name = audio_file_path.split('/')[-1]
            image_file = f"augmentations/spectrogram_{audio_name}_{i}.png"
            image.save(image_file)
            audio_save_path = f"augmentations/{task_name}/audio_{audio_name}_{i}.wav"
            if task_name == "watkins":
                species_name = audio_file_path.split('/')[-2]
                if not os.path.exists(f"augmentations/{task_name}/{species_name}"):
                    os.makedirs(f"augmentations/{task_name}/{species_name}")
                audio_save_path = f"augmentations/{task_name}/{species_name}/audio_{audio_name}_{i}.wav"
            audio_save_path_og = f"augmentations/{task_name}/{species_name}/audio_{audio_name}_{i}_og.wav"
            wavfile.write(audio_save_path, sample_rate, audio)
            wavfile.write(audio_save_path_og, sample_rate, audio_file)
    
if __name__ == "__main__":
    generate_variants(model_id = "models/watkins_train_128_ddim_5e-5", start_step=48, input_dir="../beans/data/watkins",
                        metadata_path="../beans/data/watkins/annotations.train.csv", task_name = "watkins")
