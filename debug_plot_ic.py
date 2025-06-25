import torch
import matplotlib.pyplot as plt
import numpy as np
from gray_scott_pinn import GrayScottPINN
import os

# Try both model filenames and architectures
model_configs = [
    ('gray_scott_pinn_model.pth', {'hidden_size': 100, 'num_layers': 5}),
    ('gray_scott_pinn_model_fast.pth', {'hidden_size': 50, 'num_layers': 4}),
]
model = None
for fname, config in model_configs:
    if os.path.exists(fname):
        model = GrayScottPINN(**config)
        model.load_state_dict(torch.load(fname, map_location='cpu', weights_only=True))
        model.eval()
        print(f"Loaded model: {fname} with config {config}")
        break
if model is None:
    raise FileNotFoundError("No trained model found. Please train and save a model as 'gray_scott_pinn_model.pth' or 'gray_scott_pinn_model_fast.pth'.")

# Create a uniform grid
n_per_dim = 64
x = torch.linspace(0, 1, n_per_dim)
y = torch.linspace(0, 1, n_per_dim)
X, Y = torch.meshgrid(x, y, indexing='ij')
X_flat = X.reshape(-1, 1)
Y_flat = Y.reshape(-1, 1)

# Fixed F, k for IC (should match training IC)
F = torch.full_like(X_flat, 0.055)
k = torch.full_like(X_flat, 0.062)
t = torch.zeros_like(X_flat)

# Evaluate model at t=0
with torch.no_grad():
    u_pred, v_pred = model(X_flat, Y_flat, t, F, k)
    u_pred = u_pred.numpy().reshape(n_per_dim, n_per_dim)
    v_pred = v_pred.numpy().reshape(n_per_dim, n_per_dim)

# Compute target IC (same as in gray_scott_pinn.py)
def compute_ic_targets(X, Y):
    # Standard Gray-Scott initial condition:
    # Trivial state: u = 1, v = 0 everywhere
    # Small square in center: u = 0.5, v = 0.25
    
    center_x, center_y = 0.5, 0.5
    square_size = 0.1  # Square from 0.45 to 0.55
    
    # Create mask for center square
    in_square = ((X >= center_x - square_size/2) & (X <= center_x + square_size/2) & 
                (Y >= center_y - square_size/2) & (Y <= center_y + square_size/2))
    
    # Set target values
    u_target = np.where(in_square, 0.5, 1.0)
    v_target = np.where(in_square, 0.25, 0.0)
    
    return u_target, v_target

u_target, v_target = compute_ic_targets(X.numpy(), Y.numpy())

# Plot model output and target IC side by side
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.title('Model output: Species U at t=0')
plt.imshow(u_pred, origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title('Target IC: Species U')
plt.imshow(u_target, origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title('Model output: Species V at t=0')
plt.imshow(v_pred, origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title('Target IC: Species V')
plt.imshow(v_target, origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()

plt.tight_layout()
plt.show() 