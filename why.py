import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- Path to your generated .h5 file ---
h5_path = r"C:\HyperObjectData\results\sample_001.h5"

# --- Load the cube ---
with h5py.File(h5_path, 'r') as f:
    cube = f['cube'][()]  # shape (H, W, 61)

print("Cube shape:", cube.shape)

# --- Show one spectral band (e.g., band 30) ---
band = 30
plt.imshow(cube[:, :, band], cmap='gray')
plt.title(f"Spectral Band {band}")
plt.axis('off')
plt.show()

# --- Create an approximate RGB composite ---
# Assuming channels roughly correspond to visible bands
r_band, g_band, b_band = 10, 30, 50
rgb = np.stack([
    cube[:, :, r_band],
    cube[:, :, g_band],
    cube[:, :, b_band]
], axis=-1)

# Normalize for display
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.imshow(rgb)
plt.title("Approximate RGB Composite")
plt.axis('off')
plt.show()
