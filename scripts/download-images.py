import os
import json
import argparse
from urllib.parse import urlparse, unquote

import tqdm
import magic
import requests


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"


def main(args):
    with open(args.input_file) as input_file:
        for row_json in (pbar_ds := tqdm.tqdm(input_file, position=0, desc="ID")):
            parsed = json.loads(row_json)
            pbar_ds.set_description(parsed["id"])

            save_dir = os.path.abspath(os.path.join(args.output_dir, parsed["id"]))
            os.makedirs(save_dir, exist_ok=True)

            for image_url in (
                pbar_images := tqdm.tqdm(
                    parsed["images"], position=1, leave=False, desc="Image"
                )
            ):
                filename = unquote(urlparse(image_url).path.split("/")[-1])
                filepath = os.path.join(save_dir, filename)

                pbar_images.set_description(filename)

                if os.path.exists(filepath):
                    if "image" in magic.from_file(filepath, mime=True):
                        continue

                with open(filepath, "wb") as output_file:
                    res = requests.get(
                        image_url,
                        stream=True,
                        headers={"User-Agent": USER_AGENT, "Referer": parsed["url"]},
                    )

                    for chunk in res.iter_content(chunk_size=16 * 1024):
                        output_file.write(chunk)


if __name__ == "__main__":
    # This script is used to download images from a JSON file containing house data.
    # If you're using the latest scraper, this script is not needed.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input JSON file",
        default="../data/rumah123/houses-20k.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for images",
        default="../data/rumah123/images",
    )

    args = parser.parse_args()
    main(args)
