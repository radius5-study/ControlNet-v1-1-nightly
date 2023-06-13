import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path

def bucket(_id: str) -> str:
    return _id[-3:].zfill(4)

class MyDataset(Dataset):
    def __init__(self):
        self.root = Path("/mnt/d")
        self.prompts = dict()
        self.stems = list()
        for p in (self.root/"whitewaisttag").glob("**/*.txt"):
            self.stems.append(p.stem)
            with open(p, "rt") as f:
                self.prompts[p.stem] = f.read()

        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        prompt = self.prompts[stem]

        lineart = cv2.imread(
            str(self.root / 'whitewaist_sim' / bucket(stem) / f'{stem}.png'),
            cv2.IMREAD_GRAYSCALE
        )[:, :, None]
        flatcolor = cv2.imread(
            str(self.root / 'whitewaist_flatten' / bucket(stem) / f'{stem}.png')
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

