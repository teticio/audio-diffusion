import argparse
import io
import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value, load_from_disk
from tqdm.auto import tqdm

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("add_cond_prev")


def main(args):
    dataset = load_from_disk(args.input_dataset)["train"]

    audio_file_map = {}
    for entry in tqdm(dataset):
        audio_file = entry["audio_file"]
        slice_value = entry["slice"]
        image = entry["image"]

        if audio_file not in audio_file_map:
            audio_file_map[audio_file] = {}

        audio_file_map[audio_file][slice_value] = image

    new_data = []
    for entry in tqdm(dataset):
        audio_file = entry["audio_file"]
        slice_value = entry["slice"]

        cond_prev = audio_file_map[audio_file].get(slice_value - 1, None)
        if cond_prev:
            with io.BytesIO() as output:
                entry["image"].save(output, format="PNG")
                image_bytes = output.getvalue()
            with io.BytesIO() as output:
                cond_prev.save(output, format="PNG")
                cond_prev_bytes = output.getvalue()
            new_data.append(
                {
                    "image": {"bytes": image_bytes},
                    "cond_prev": {"bytes": cond_prev_bytes},
                    "audio_file": audio_file,
                    "slice": slice_value,
                }
            )

    new_dataset = Dataset.from_pandas(
        pd.DataFrame(new_data),
        Features(
            {
                "image": Image(),
                "cond_prev": Image(),
                "audio_file": Value(dtype="string"),
                "slice": Value(dtype="int16"),
            }
        ),
    )
    dsd = DatasetDict({"train": new_dataset})
    dsd.save_to_disk(os.path.join(args.output_dataset))
    if args.push_to_hub:
        dsd.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add cond_prev column to existing dataset.")
    parser.add_argument("--input_dataset", type=str)
    parser.add_argument("--output_dataset", type=str)
    parser.add_argument("--push_to_hub", type=str, default=None)
    args = parser.parse_args()

    main(args)
