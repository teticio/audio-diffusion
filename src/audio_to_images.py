import os
import re
import json
import argparse

from tqdm.auto import tqdm

from mel import Mel


def main(args):
    mel = Mel(x_res=args.resolution, y_res=args.resolution)
    os.makedirs(args.output_dir, exist_ok=True)
    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.input_dir)
        for file in files
        if re.search("\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    meta_data = {}
    try:
        for audio, audio_file in enumerate(tqdm(audio_files)):
            try:
                mel.load_audio(audio_file)
            except KeyboardInterrupt:
                raise
            except:
                continue
            for slice in range(mel.get_number_of_slices()):
                image = mel.audio_slice_to_image(slice)
                image_file = f"{audio}_{slice}.png"
                image.save(os.path.join(args.output_dir, image_file))
                meta_data[image_file] = audio_file
    finally:
        with open(os.path.join(args.output_dir, 'meta_data.json'), 'wt') as file:
            file.write(json.dumps(meta_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio into Mel spectrograms.")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    main(args)
