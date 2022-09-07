import argparse

import gradio as gr

from audiodiffusion import AudioDiffusion


def generate_spectrogram_audio_and_loop(model_id):
    audio_diffusion = AudioDiffusion(model_id=model_id)
    image, (sample_rate,
            audio) = audio_diffusion.generate_spectrogram_and_audio()
    loop = AudioDiffusion.loop_it(audio, sample_rate)
    if loop is None:
        loop = audio
    return image, (sample_rate, audio), (sample_rate, loop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--server", type=int)
    args = parser.parse_args()

    demo = gr.Interface(
        fn=generate_spectrogram_audio_and_loop,
        title="Audio Diffusion",
        description="Generate audio using Huggingface diffusers.\
            This takes about 20 minutes without a GPU, so why not make yourself a cup of tea in the meantime?",
        inputs=[
            gr.Dropdown(label="Model",
                        choices=[
                            "teticio/audio-diffusion-256",
                            "teticio/audio-diffusion-breaks-256"
                        ],
                        value="teticio/audio-diffusion-256")
        ],
        outputs=[
            gr.Image(label="Mel spectrogram", image_mode="L"),
            gr.Audio(label="Audio"),
            gr.Audio(label="Loop"),
        ],
        allow_flagging="never"
    )
    demo.launch(server_name=args.server or "0.0.0.0", server_port=args.port)
