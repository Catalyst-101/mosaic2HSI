import os, glob

def list_mosaic_files(root):
    return sorted(glob.glob(os.path.join(root, 'mosaic', '*.npy')))

def list_hsi_files(root):
    return sorted(glob.glob(os.path.join(root, 'hsi_61', '*.h5')))
