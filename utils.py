import os
import torch

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved -> {path}")

def load_checkpoint(model, path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded checkpoint from {path}")
    return model
