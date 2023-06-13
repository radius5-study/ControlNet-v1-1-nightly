from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import config
from cldm.hack import hack_everything
hack_everything(clip_skip=2)

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.lineart_anime import LineartAnimeDetector
from cldm.ddim_hacked import DDIMSampler

# Configs
#resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model_name = 'control_v11p_sd15s2_lineart_anime'
base_model = 'anything-v3-full.safetensors'
base_model = 'ACertainThing.ckpt'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict(f'./models/{base_model}', location='cpu'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

trainer = pl.Trainer(gpus=1, precision=32, callbacks=[])

wandb_logger = WandbLogger()
tb_logger = TensorBoardLogger("tb_logs")

checkpoint_callback = ModelCheckpoint(
    every_n_epochs=1,
    monitor=None,
    dirpath="./output",
    filename="seniro-{epoch}",
    save_top_k=-1,
    save_last=True,
    verbose=True,
)

trainer = pl.Trainer(
    gpus=1,
    precision=32,
    logger=[wandb_logger, tb_logger],
    callbacks=[checkpoint_callback],
)


# Train!
trainer.fit(model, dataloader)
