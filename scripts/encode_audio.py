import argparse
import os
import pickle

from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm

from audiodiffusion.audio_encoder import AudioEncoder


def main(args):
    audio_encoder = AudioEncoder.from_pretrained("teticio/audio-encoder")

    if args.dataset_name is not None:
        if os.path.exists(args.dataset_name):
            dataset = load_from_disk(args.dataset_name)["train"]
        else:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                use_auth_token=True if args.use_auth_token else None,
                split="train",
            )

    encodings = {}
    for audio_file in tqdm(dataset.to_pandas()["audio_file"].unique()):
        encodings[audio_file] = audio_encoder.encode([audio_file])
    pickle.dump(encodings, open(args.output_file, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create pickled audio encodings for dataset of audio files.")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="data/encodings.p")
    parser.add_argument("--use_auth_token", type=bool, default=False)
    args = parser.parse_args()
    main(args)
