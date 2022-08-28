import argparse

import gradio as gr

from audiodiffusion import AudioDiffusion

audio_diffusion = AudioDiffusion()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--server", type=int)
    args = parser.parse_args()

    demo = gr.Interface(
        fn=audio_diffusion.generate_spectrogram_and_audio,
        title="Audio Diffusion",
        description="Generate audio using Huggingface diffusers.\
            This takes about 20 minutes without a GPU, so why not make yourself a cup of tea in the meantime?",
        inputs=[],
        outputs=[
            gr.Image(label="Mel spectrogram", image_mode="L"),
            gr.Audio(label="Audio"),
        ],
    )
    demo.launch(server_name=args.server or "0.0.0.0", server_port=args.port)
