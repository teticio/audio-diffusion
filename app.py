import argparse

import gradio as gr

from audiodiffusion import AudioDiffusion


def generate_spectrogram_audio_and_loop(model_id):
    audio_diffusion = AudioDiffusion(model_id=model_id)
    image, (sample_rate, audio) = audio_diffusion.generate_spectrogram_and_audio()
    loop = AudioDiffusion.loop_it(audio, sample_rate)
    if loop is None:
        loop = audio
    return image, (sample_rate, audio), (sample_rate, loop)


demo = gr.Interface(
    fn=generate_spectrogram_audio_and_loop,
    title="Audio Diffusion",
    description="Generate audio using Huggingface diffusers.\
        The models without 'latent' or 'ddim' give better results but take about \
            20 minutes without a GPU. For GPU, you can use \
                [colab](https://colab.research.google.com/github/teticio/audio-diffusion/blob/master/notebooks/gradio_app.ipynb) \
                    to run this app.",
    inputs=[
        gr.Dropdown(
            label="Model",
            choices=[
                "teticio/audio-diffusion-256",
                "teticio/audio-diffusion-breaks-256",
                "teticio/audio-diffusion-instrumental-hiphop-256",
                "teticio/audio-diffusion-ddim-256",
                "teticio/latent-audio-diffusion-256",
                "teticio/latent-audio-diffusion-ddim-256",
            ],
            value="teticio/latent-audio-diffusion-ddim-256",
        )
    ],
    outputs=[
        gr.Image(label="Mel spectrogram", image_mode="L"),
        gr.Audio(label="Audio"),
        gr.Audio(label="Loop"),
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--server", type=int)
    args = parser.parse_args()
    demo.launch(server_name=args.server or "0.0.0.0", server_port=args.port)
