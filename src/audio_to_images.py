import os
import re
import io
import argparse

import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, Features, Image, Value

from mel import Mel


def main(args):
    mel = Mel(x_res=args.resolution, y_res=args.resolution, hop_length=args.hop_length)
    os.makedirs(args.output_dir, exist_ok=True)
    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.input_dir)
        for file in files
        if re.search("\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    examples = []
    try:
        for audio_file in tqdm(audio_files):
            try:
                mel.load_audio(audio_file)
            except KeyboardInterrupt:
                raise
            except:
                continue
            for slice in range(mel.get_number_of_slices()):
                image = mel.audio_slice_to_image(slice)
                assert (
                    image.width == args.resolution and image.height == args.resolution
                )
                with io.BytesIO() as output:
                    image.save(output, format="PNG")
                    bytes = output.getvalue()
                examples.extend(
                    [
                        {
                            "image": {"bytes": bytes},
                            "audio_file": audio_file,
                            "slice": slice,
                        }
                    ]
                )
    finally:
        ds = Dataset.from_pandas(
            pd.DataFrame(examples),
            features=Features(
                {
                    "image": Image(),
                    "audio_file": Value(dtype="string"),
                    "slice": Value(dtype="int16"),
                }
            ),
        )
        dsd = DatasetDict({"train": ds})
        dsd.save_to_disk(os.path.join(args.output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create dataset of Mel spectrograms from directory of audio files."
    )
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--hop_length", type=int, default=512)
    args = parser.parse_args()
    main(args)
