import os
import glob
import random
import numpy as np
import h5py
from torch.utils.data import Dataset


def list_files(root_dir):
    # looks for mosaic .npy and matching h5 in hsi_61
    mosaic_dir = os.path.join(root_dir, 'mosaic')
    hsi_dir = os.path.join(root_dir, 'hsi_61')
    mosaics = sorted(glob.glob(os.path.join(mosaic_dir, '*.npy')))
    out = []
    for m in mosaics:
        name = os.path.splitext(os.path.basename(m))[0]
        h5_path = os.path.join(hsi_dir, name + '.h5')
        if os.path.exists(h5_path):
            out.append((m, h5_path))
    return out


class HSIDataset(Dataset):
    """Dataset yields patches (mosaic_patch, hsi_patch).
    mosaic: (H,W) float32, hsi: (H,W,61) float32
    Returns tensors shaped: (C_in, H, W), (C_out, H, W)
    """
    def __init__(self, root_dir, patch_size=256, is_train=True, transforms=None):
        self.pairs = list_files(root_dir)
        self.patch_size = patch_size
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mosaic_path, h5_path = self.pairs[idx]
        mosaic = np.load(mosaic_path).astype('float32')  # (H,W)
        with h5py.File(h5_path, 'r') as f:
            cube = f['cube'][()]  # (H,W,61)
            # optional: read wavelengths = f['wavelengths'][()]

        H, W = mosaic.shape
        ps = self.patch_size
        if self.is_train:
            # sample a random patch
            i = random.randint(0, H - ps)
            j = random.randint(0, W - ps)
            mosaic_patch = mosaic[i:i+ps, j:j+ps]
            hsi_patch = cube[i:i+ps, j:j+ps, :]
        else:
            # center crop for eval, or full-size later in inference
            i = (H - ps) // 2
            j = (W - ps) // 2
            mosaic_patch = mosaic[i:i+ps, j:j+ps]
            hsi_patch = cube[i:i+ps, j:j+ps, :]

        # normalize if needed (already 0-1)
        # to CHW
        mosaic_patch = mosaic_patch[None, :, :]
        hsi_patch = np.transpose(hsi_patch, (2,0,1))

        return mosaic_patch, hsi_patch