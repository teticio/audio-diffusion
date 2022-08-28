from io import BytesIO
import streamlit as st
import soundfile as sf
from librosa.util import normalize

from audiodiffusion import AudioDiffusion

audio_diffusion = AudioDiffusion()

if __name__ == "__main__":
    st.header("Audio Diffusion")
    st.markdown("Generate audio using Huggingface diffusers.\
        This takes about 20 minutes without a GPU, so why not make yourself a cup of tea in the meantime?"
                )
    if st.button("Generate"):
        st.markdown("Generating...")
        image, (sample_rate,
                audio) = audio_diffusion.generate_spectrogram_and_audio()
        st.image(image, caption="Mel spectrogram")
        buffer = BytesIO()
        sf.write(buffer, normalize(audio), sample_rate, format="WAV")
        st.audio(buffer, format="audio/wav")
