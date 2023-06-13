# %%
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
from PIL import Image

# Configs
#resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# %%

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
#model_name = 'control_v11p_sd15s2_lineart_anime'
model_name = 'seniro'
base_model = 'anything-v3-full.safetensors'
base_model = 'ACertainThing.ckpt'
# %%
state_dict = load_state_dict(f'./models/{model_name}.pth', location='cpu')
key = "control_model.input_hint_block.0.weight"
state_dict[key] = torch.cat((state_dict[key], state_dict[key][:, 0:1, :, :]), dim=1)
torch.save(state_dict, f'./models/{model_name}2.pth')
# %%
print(state_dict[key].shape)

model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict(f'./models/{base_model}', location='cpu'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
model.to(device="cuda")
ddim_sampler = DDIMSampler(model)
# %%
# input_hint_blockを探す
state_dict = load_state_dict(f'./models/{model_name}.pth', location='cpu')
# %%
for k, v in state_dict.items():
    if 'hint' in k:
        print(k)
# control_model.input_hint_block.0.weight 16, 3, 3, 3
# control_model.input_hint_block.0.bias, 16
# config["params"]["control_stage_config"]["params"]["hint_channels"]
# config["model"]["params"]["control_stage_config"]["params"]["hint_channels"]

# %%
filepath = "/mnt/c/Users/qzrp0/Downloads/0_0_lineart.jpg"
input_image = np.array(Image.open(filepath).convert('RGB')).astype(np.uint8)

image_resolution = 512
img = resize_image(input_image, image_resolution)
H, W, C = img.shape

detected_map = input_image.copy()
detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

num_samples = 1
control = 1.0 - torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
control = torch.stack([control for _ in range(num_samples)], dim=0)
control = einops.rearrange(control, 'b h w c -> b c h w').clone()
control = torch.zeros([1, 4, 1024, 512], dtype=torch.float32, device="cuda")
prompt = ""
a_prompt = "masterpiece, best quality, ultra-detailed, illustration, disheveled hair"
n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair,extra digit, fewer digits, cropped, worst quality, low quality"
strength = 1.0
ddim_steps = 10 # 20
shape = (4, 128, 64)
eta = 1
scale = 9
cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
model.control_scales = [strength] * 13
samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, cond, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=un_cond)

x_samples = model.decode_first_stage(samples)
x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

results = [x_samples[i] for i in range(num_samples)]
# %%
Image.fromarray(results[0])

# RuntimeError: Given groups=1, weight of size [16, 3, 3, 3], expected input[1, 4, 1024, 512] to have 3 channels, but got 4 channels instead