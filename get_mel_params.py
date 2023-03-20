FFT_SIZE_IN_SECS = 0.05
HOP_LENGTH_IN_SECS = 0.025
SAMPLE_RATE = 44100

def get_mel_params(downstream_max_duration):
    max_frames = 128 # x
    n_mels = 128 # y
    n_fft = int(SAMPLE_RATE * FFT_SIZE_IN_SECS)
    # hop_length = int(SAMPLE_RATE * HOP_LENGTH_IN_SECS)
    

    
    needed_frames_per_sec = max_frames / downstream_max_duration - 1
    needed_hop_length = int(1 / needed_frames_per_sec * SAMPLE_RATE)
    max_duration = max_frames / needed_frames_per_sec
    print(f"n_fft is {n_fft} and hop length is {needed_hop_length}")
    print("max duration", max_duration)
    assert downstream_max_duration < max_duration
    
    
if __name__ == "__main__":
    get_mel_params(3)