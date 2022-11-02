from io import BytesIO
import streamlit as st
import soundfile as sf
from librosa.util import normalize
from librosa.beat import beat_track

from audiodiffusion import AudioDiffusion

if __name__ == "__main__":
    st.header("Audio Diffusion")
    st.markdown("Generate audio using Huggingface diffusers.\
        This takes about 20 minutes without a GPU, so why not make yourself a \
            cup of tea in the meantime? (Or try the teticio/audio-diffusion-ddim-256 \
                model which is faster.)")

    model_id = st.selectbox("Model", [
        "teticio/audio-diffusion-256", "teticio/audio-diffusion-breaks-256",
        "teticio/audio-diffusion-instrumental-hiphop-256",
        "teticio/audio-diffusion-ddim-256"
    ])
    audio_diffusion = AudioDiffusion(model_id=model_id)

    if st.button("Generate"):
        st.markdown("Generating...")
        image, (sample_rate,
                audio) = audio_diffusion.generate_spectrogram_and_audio()
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
