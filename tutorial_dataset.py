import cv2
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path
import csv
from itertools import product
from scipy.ndimage import grey_erosion

def bucket(_id: str) -> str:
    return _id[-3:].zfill(4)

class MyDataset(Dataset):
    def __init__(self):
        self.root = Path("/mnt/d")
        self.csv_path = self.root/"filepath.csv"
        self.prompts = dict()
        self.stems = list()
        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)
            for path in reader:
                p = Path(path[0]).stem
                with (self.root/"whitewaist_tag"/bucket(p)/f"{p}.txt").open("rt") as f:
                    self.prompts[p] = f.read()
                self.stems.append(p)
        self.flatcolor_choices = ['whitewaist_flatten', 'whitewaist_slic1', 'whitewaist_slic30']
        self.lineart_choices = ['whitewaist_lineart_anime', 'whitewaist_lineart', 'whitewaist_sketch', 'whitewaist_sim']
        self.erosion_choices = [1, 2, 3]
        self.state_product = list(product(self.lineart_choices, self.flatcolor_choices, self.erosion_choices))
        np.random.RandomState(0).shuffle(self.state_product)
        self.random_state = np.random.RandomState(0).randint(len(self.state_product), size=len(self.stems))

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        lineart_choice, flatcolor_choice, erosion_choice = self.state_product[self.random_state[idx]]
        self.random_state[idx] = (self.random_state[idx]+1)%len(self.state_product)
        stem = self.stems[idx]
        prompt = self.prompts[stem]

        lineart = cv2.imread(
            str(self.root / lineart_choice / bucket(stem) / f'{stem}.png'),
            cv2.IMREAD_GRAYSCALE
        )
        if erosion_choice == 1:
            lineart = lineart[:, :, None]
        else:
            lineart = grey_erosion(lineart, size=(erosion_choice, erosion_choice))[:, :, None]

        flatcolor = cv2.imread(
            str(self.root / flatcolor_choice / bucket(stem) / f'{stem}.png')
        )
        target = cv2.imread(
            str(self.root / 'whitewaist' / bucket(stem) / f'{stem}.png')
        )

        # Do not forget that OpenCV read images in BGR order.
        flatcolor = cv2.cvtColor(flatcolor, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        lineart = 1 - lineart.astype(np.float32) / 255.0
        flatcolor = flatcolor.astype(np.float32) / 255.0
        hint = np.concatenate([lineart, flatcolor], axis=2)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=hint)

