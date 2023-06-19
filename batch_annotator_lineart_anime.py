from annotator.util import resize_image, HWC3
from tqdm import tqdm
import argparse
from argparse import Namespace
from pathlib import Path
from typing import Union
import csv
import numpy as np
from PIL import Image

model_lineart_anime = None
def lineart_anime(img, res):
    img = resize_image(HWC3(img), res)
    global model_lineart_anime
    if model_lineart_anime is None:
        from annotator.lineart_anime import LineartAnimeDetector
        model_lineart_anime = LineartAnimeDetector()
    result = model_lineart_anime(img)
    return [result]

model_lineart = None
def lineart(img, res, coarse=False):
    img = resize_image(HWC3(img), res)
    global model_lineart
    if model_lineart is None:
        from annotator.lineart import LineartDetector
        model_lineart = LineartDetector()
    result = model_lineart(img, coarse)
    return [result]

def bucket(_id: str) -> str:
    return _id[-3:].zfill(4)

def png_to_numpy(png_path: Union[str, Path]) -> np.ndarray:
    png_image = Image.open(png_path)
    rgb_image = png_image.convert('RGB')
    numpy_array = np.array(rgb_image, dtype=np.uint8)
    return numpy_array

def main(args: Namespace) -> None:
    with open(args.csvpath, "r") as f:
        reader = csv.reader(f)
        file_paths = [Path(line[0]) for line in reader]
    for file_path in tqdm(file_paths):
        img = png_to_numpy(file_path)
        res = 512
        out = lineart_anime(img, res)
        out_path = Path(args.root)/args.savedir/f"{bucket(file_path.stem)}"/f"{file_path.stem}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out[0]).save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvpath", type=str, default="/mnt/d/filepath.csv")
    parser.add_argument("--root", type=str, default="/mnt/d")
    parser.add_argument("--savedir", type=str, default="whitewaist_lineart_anime")
    args = parser.parse_args()
    main(args)