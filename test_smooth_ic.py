#!/usr/bin/env python3
"""
Test script to verify that the smooth IC can be learned by the neural network
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class GrayScottPINN(nn.Module):
    def __init__(self, hidden_size=100, num_layers=5):
        super(GrayScottPINN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(5, hidden_size))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_size, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, y, t, F, k):
        inputs = torch.cat([x, y, t, F, k], dim=1)
        output = self.network(inputs)
        u = output[:, 0:1]
        v = output[:, 1:2]
        return u, v

def test_smooth_ic_learning():
    """Test if model can learn the new SMOOTH initial condition"""
    print("🔍 TESTING SMOOTH IC LEARNING")
    print("=" * 50)
    
    # Create grid
    n_per_dim = 64
    x = torch.linspace(0, 1, n_per_dim).reshape(-1, 1)
    y = torch.linspace(0, 1, n_per_dim).reshape(-1, 1)
    X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    t = torch.zeros_like(X_flat)
    F = torch.full_like(X_flat, 0.055)
    k = torch.full_like(X_flat, 0.062)
    
    # Create SMOOTH IC targets with 3x3 grid (same as in updated model)
    bump_centers = [
        (0.25, 0.25), (0.5, 0.25), (0.75, 0.25),  # Bottom row
        (0.25, 0.5),  (0.5, 0.5),  (0.75, 0.5),   # Middle row  
        (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)   # Top row
    ]
    
    radius = 0.06
    sigmoid_sharpness = 20.0
    
    # Initialize as base values
    u_target = torch.ones_like(X_flat)  # Base value u = 1
    v_target = torch.zeros_like(X_flat)  # Base value v = 0
    
    # Add each bump to the initial condition
    for center_x, center_y in bump_centers:
        # Distance from this bump center
        dist = torch.sqrt((X_flat - center_x) ** 2 + (Y_flat - center_y) ** 2)
        
        # Smooth bump: u decreases, v increases near center
        u_bump = -0.5 * torch.sigmoid(sigmoid_sharpness * (radius - dist))
        v_bump = 0.25 * torch.sigmoid(sigmoid_sharpness * (radius - dist))
        
        # Add this bump's contribution
        u_target = u_target + u_bump
        v_target = v_target + v_bump
    
    # Clamp to reasonable ranges
    u_target = torch.clamp(u_target, 0.4, 1.0)
    v_target = torch.clamp(v_target, 0.0, 0.3)
    
    print(f"Smooth IC Target Stats:")
    print(f"  U: min={u_target.min():.6f}, max={u_target.max():.6f}, range={u_target.max()-u_target.min():.6f}")
    print(f"  V: min={v_target.min():.6f}, max={v_target.max():.6f}, range={v_target.max()-v_target.min():.6f}")
    
    # Model
    model = GrayScottPINN(hidden_size=50, num_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        u_pred, v_pred = model(X_flat, Y_flat, t, F, k)
        loss = torch.mean((u_pred - u_target) ** 2) + torch.mean((v_pred - v_target) ** 2)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.8f}")
    
    # Evaluate
    with torch.no_grad():
        u_pred, v_pred = model(X_flat, Y_flat, t, F, k)
        print(f"\nFinal Results:")
        print(f"  U pred: min={u_pred.min():.6f}, max={u_pred.max():.6f}, range={u_pred.max()-u_pred.min():.6f}")
        print(f"  V pred: min={v_pred.min():.6f}, max={v_pred.max():.6f}, range={v_pred.max()-v_pred.min():.6f}")
        
        # Correlation
        u_corr = torch.corrcoef(torch.stack([u_pred.squeeze(), u_target.squeeze()]))[0,1]
        v_corr = torch.corrcoef(torch.stack([v_pred.squeeze(), v_target.squeeze()]))[0,1]
        print(f"  Correlation: U={u_corr:.6f}, V={v_corr:.6f}")
        
        # Check bump vs background values
        # Combine all bump regions
        bump_mask = torch.zeros_like(X_flat, dtype=torch.bool).squeeze()
        for center_x, center_y in bump_centers:
            dist_to_bump = torch.sqrt((X_flat - center_x) ** 2 + (Y_flat - center_y) ** 2)
            bump_mask = bump_mask | (dist_to_bump < radius).squeeze()
        
        background_mask = ~bump_mask
        
        print(f"\nRegion Analysis:")
        print(f"  U in bumps: mean={u_pred[bump_mask].mean():.6f} (target ≈ 0.5)")
        print(f"  U in background: mean={u_pred[background_mask].mean():.6f} (target ≈ 1.0)")
        print(f"  V in bumps: mean={v_pred[bump_mask].mean():.6f} (target ≈ 0.25)")
        print(f"  V in background: mean={v_pred[background_mask].mean():.6f} (target ≈ 0.0)")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # U comparison
    im1 = axes[0,0].imshow(u_target.reshape(n_per_dim, n_per_dim).numpy(), 
                          origin='lower', extent=[0,1,0,1], vmin=0.4, vmax=1.0)
    axes[0,0].set_title('U Target (3x3 Bumps)')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(u_pred.reshape(n_per_dim, n_per_dim).numpy(), 
                          origin='lower', extent=[0,1,0,1], vmin=0.4, vmax=1.0)
    axes[0,1].set_title('U Predicted')
    plt.colorbar(im2, ax=axes[0,1])
    
    # V comparison
    im3 = axes[1,0].imshow(v_target.reshape(n_per_dim, n_per_dim).numpy(), 
                          origin='lower', extent=[0,1,0,1], vmin=0.0, vmax=0.3)
    axes[1,0].set_title('V Target (3x3 Bumps)')
    plt.colorbar(im3, ax=axes[1,0])
    
    im4 = axes[1,1].imshow(v_pred.reshape(n_per_dim, n_per_dim).numpy(), 
                          origin='lower', extent=[0,1,0,1], vmin=0.0, vmax=0.3)
    axes[1,1].set_title('V Predicted')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.show()
    
    return u_corr.item(), v_corr.item()

if __name__ == "__main__":
    print("🧠 Testing Smooth IC Learning for Gray-Scott PINN")
    print("=" * 60)
    
    u_corr, v_corr = test_smooth_ic_learning()
    
    print("\n" + "=" * 60)
    print("📊 SMOOTH IC LEARNING RESULTS")
    print("=" * 60)
    print(f"✅ U correlation: {u_corr:.6f} {'(EXCELLENT)' if u_corr > 0.95 else '(GOOD)' if u_corr > 0.8 else '(POOR)'}")
    print(f"✅ V correlation: {v_corr:.6f} {'(EXCELLENT)' if v_corr > 0.95 else '(GOOD)' if v_corr > 0.8 else '(POOR)'}")
    
    if u_corr > 0.8 and v_corr > 0.8:
        print("\n🎉 SUCCESS! The smooth IC can be learned by the neural network!")
        print("   → This should fix the IC learning problem in the main PINN")
        print("   → The smooth sigmoid transitions are neural network friendly")
    else:
        print("\n❌ Still having issues - may need further adjustment")
        print("   → Try reducing sigmoid_sharpness for smoother transitions")
        print("   → Or increase model capacity (more neurons/layers)") 