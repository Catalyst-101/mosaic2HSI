import os
import h5py
import numpy as np
import torch
from models.unet import UNet


def infer_full(mosaic_path, ckpt_path, out_h5_path, device='cpu'):
    mosaic = np.load(mosaic_path).astype('float32')  # (H,W)
    H, W = mosaic.shape
    model = UNet(in_channels=1, out_channels=61)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    # Sliding-window inference
    ps = 256
    stride = 256
    pred_cube = np.zeros((61, H, W), dtype='float32')
    count = np.zeros((H, W), dtype='float32')

    with torch.no_grad():
        for i in range(0, H - ps + 1, stride):
            for j in range(0, W - ps + 1, stride):
                patch = mosaic[i:i+ps, j:j+ps][None, None, ...]  # (1,1,ps,ps)
                inp = torch.from_numpy(patch).to(device)
                out = model(inp).cpu().numpy()[0]  # (61,ps,ps)
                pred_cube[:, i:i+ps, j:j+ps] += out
                count[i:i+ps, j:j+ps] += 1

    # Handle any areas not covered (edges)
    count[count == 0] = 1
    pred_cube /= count[None, :, :]

    # Transpose to (H, W, 61)
    pred_cube = np.transpose(pred_cube, (1, 2, 0))

    os.makedirs(os.path.dirname(out_h5_path), exist_ok=True)
    with h5py.File(out_h5_path, 'w') as f:
        f.create_dataset('cube', data=pred_cube.astype('float32'), compression='gzip')

    print(f"✅ Saved reconstructed cube to {out_h5_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python infer.py <mosaic.npy> <checkpoint.pth> <output.h5>")
        exit(1)
    _, mosaic_path, ckpt_path, out_h5_path = sys.argv
    infer_full(mosaic_path, ckpt_path, out_h5_path)
