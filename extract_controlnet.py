import argparse
import torch
from cldm.model import load_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--dst", default=None, type=str, required=False, help="Path to the output model.")
    args = parser.parse_args()
    assert args.src is not None, "Must provide a model path!"
    if args.dst is None:
        args.dst = ".".join(args.src.split(".")[:-1]) + ".extract.pth"
    state_dict = load_state_dict(args.src, location='cpu')
    for k in list(state_dict.keys()):
        if not k.startswith('control_model.'):
            state_dict.pop(k)
    torch.save(state_dict, args.dst)
    