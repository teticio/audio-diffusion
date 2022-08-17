import argparse

import gradio as gr
from PIL import Image
from diffusers import DDPMPipeline

from src.mel import Mel

mel = Mel(x_res=256, y_res=256)
model_id = "teticio/audio-diffusion-256"
ddpm = DDPMPipeline.from_pretrained(model_id)


def generate_spectrogram_and_audio():
    images = ddpm(output_type="numpy")["sample"]
    images = (images * 255).round().astype("uint8").transpose(0, 3, 1, 2)
    image = Image.fromarray(images[0][0])
    audio = mel.image_to_audio(image)
    return image, (mel.get_sample_rate(), audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--server", type=int)
    args = parser.parse_args()

    demo = gr.Interface(
        fn=generate_spectrogram_and_audio,
        title="Audio Diffusion",
        description=f"Generate audio using Huggingface diffusers.\
            This takes about 20 minutes without a GPU, so why not make yourself a cup of tea in the meantime?",
        inputs=[],
        outputs=[
            gr.Image(label="Mel spectrogram", image_mode="L"),
            gr.Audio(label="Audio"),
        ],
    )
    demo.launch(server_name=args.server or "0.0.0.0", server_port=args.port)
