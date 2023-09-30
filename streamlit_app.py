from io import BytesIO

import soundfile as sf
import streamlit as st
from librosa.beat import beat_track
from librosa.util import normalize

from audiodiffusion import AudioDiffusion

if __name__ == "__main__":
    st.header("Audio Diffusion")
    st.markdown(
        "Generate audio using Huggingface diffusers.\
        The models without 'latent' or 'ddim' give better results but take about \
            20 minutes without a GPU.",
    )

    model_id = st.selectbox(
        "Model",
        [
            "teticio/audio-diffusion-256",
            "teticio/audio-diffusion-breaks-256",
            "teticio/audio-diffusion-instrumental-hiphop-256",
            "teticio/audio-diffusion-ddim-256",
            "teticio/latent-audio-diffusion-256",
            "teticio/latent-audio-diffusion-ddim-256",
        ],
        index=5,
    )
    audio_diffusion = AudioDiffusion(model_id=model_id)

    if st.button("Generate"):
        st.markdown("Generating...")
        image, (sample_rate, audio) = audio_diffusion.generate_spectrogram_and_audio()
        st.image(image, caption="Mel spectrogram")
        buffer = BytesIO()
        sf.write(buffer, normalize(audio), sample_rate, format="WAV")
        st.audio(buffer, format="audio/wav")

        audio = AudioDiffusion.loop_it(audio, sample_rate)
        if audio is not None:
            st.markdown("Loop")
            buffer = BytesIO()
            sf.write(buffer, normalize(audio), sample_rate, format="WAV")
            st.audio(buffer, format="audio/wav")
