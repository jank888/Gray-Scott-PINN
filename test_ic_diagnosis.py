#!/usr/bin/env python3
"""
Diagnostic script to identify why IC learning fails in Gray-Scott PINN
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
        
        self.Du = 0.16
        self.Dv = 0.08
    
    def forward(self, x, y, t, F, k):
        inputs = torch.cat([x, y, t, F, k], dim=1)
        output = self.network(inputs)
        u = output[:, 0:1]
        v = output[:, 1:2]
        return u, v

def test_ic_only_training():
    """Test training with ONLY IC loss to isolate the problem"""
    print("🔍 TESTING IC-ONLY TRAINING")
    print("=" * 50)
    
    # Create the exact same setup as main training
    n_per_dim = int(np.sqrt(2000))  # ~45x45 = 2025 points
    x_grid = torch.linspace(0, 1, n_per_dim)
    y_grid = torch.linspace(0, 1, n_per_dim)
    X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
    X_spatial = X_grid.reshape(-1, 1)
    Y_spatial = Y_grid.reshape(-1, 1)
    
    # IC points: same spatial locations, t=0, fixed F, k
    x_ic, y_ic = X_spatial, Y_spatial
    t_ic = torch.zeros_like(X_spatial)
    F_ic = torch.full_like(X_spatial, 0.055)
    k_ic = torch.full_like(X_spatial, 0.062)
    
    # Create IC targets (exact same logic as main training)
    center_x, center_y = 0.5, 0.5
    square_size = 0.1
    in_square = ((x_ic >= center_x - square_size/2) & (x_ic <= center_x + square_size/2) & 
                (y_ic >= center_y - square_size/2) & (y_ic <= center_y + square_size/2))
    u_target_ic = torch.where(in_square, torch.tensor(0.5), torch.tensor(1.0))
    v_target_ic = torch.where(in_square, torch.tensor(0.25), torch.tensor(0.0))
    
    print(f"IC Target Stats:")
    print(f"  U: min={u_target_ic.min():.6f}, max={u_target_ic.max():.6f}, range={u_target_ic.max()-u_target_ic.min():.6f}")
    print(f"  V: min={v_target_ic.min():.6f}, max={v_target_ic.max():.6f}, range={v_target_ic.max()-v_target_ic.min():.6f}")
    print(f"  Training points: {len(x_ic)}")
    print()
    
    # Model (same as main training)
    model = GrayScottPINN(hidden_size=50, num_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop - ONLY IC loss
    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # ONLY IC loss (exact same computation as main training)
        u_ic_pred, v_ic_pred = model(x_ic, y_ic, t_ic, F_ic, k_ic)
        ic_loss = torch.mean((u_ic_pred - u_target_ic) ** 2) + torch.mean((v_ic_pred - v_target_ic) ** 2)
        
        ic_loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, IC Loss: {ic_loss.item():.8f}")
    
    # Evaluate
    with torch.no_grad():
        u_pred, v_pred = model(x_ic, y_ic, t_ic, F_ic, k_ic)
        print(f"\nFinal Results:")
        print(f"  U pred: min={u_pred.min():.6f}, max={u_pred.max():.6f}, range={u_pred.max()-u_pred.min():.6f}")
        print(f"  V pred: min={v_pred.min():.6f}, max={v_pred.max():.6f}, range={v_pred.max()-v_pred.min():.6f}")
        
        # Check correlation
        u_corr = torch.corrcoef(torch.stack([u_pred.squeeze(), u_target_ic.squeeze()]))[0,1]
        v_corr = torch.corrcoef(torch.stack([v_pred.squeeze(), v_target_ic.squeeze()]))[0,1]
        print(f"  Correlation: U={u_corr:.6f}, V={v_corr:.6f}")
        
        # Check step learning specifically
        u_in_square = u_pred[in_square.squeeze()]
        u_out_square = u_pred[~in_square.squeeze()]
        v_in_square = v_pred[in_square.squeeze()]
        v_out_square = v_pred[~in_square.squeeze()]
        
        print(f"  U inside square: mean={u_in_square.mean():.6f}, std={u_in_square.std():.6f} (target: 0.5)")
        print(f"  U outside square: mean={u_out_square.mean():.6f}, std={u_out_square.std():.6f} (target: 1.0)")
        print(f"  V inside square: mean={v_in_square.mean():.6f}, std={v_in_square.std():.6f} (target: 0.25)")
        print(f"  V outside square: mean={v_out_square.mean():.6f}, std={v_out_square.std():.6f} (target: 0.0)")
    
    return u_corr.item(), v_corr.item()

def test_step_function_learning():
    """Test if model can learn a simple step function"""
    print("\n🔍 TESTING STEP FUNCTION LEARNING")
    print("=" * 50)
    
    # Same grid as other tests
    n_per_dim = 64
    x = torch.linspace(0, 1, n_per_dim).reshape(-1, 1)
    y = torch.linspace(0, 1, n_per_dim).reshape(-1, 1)
    X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    t = torch.zeros_like(X_flat)
    F = torch.full_like(X_flat, 0.055)
    k = torch.full_like(X_flat, 0.062)
    
    # Simple step function: u=1 everywhere, v=0.25 in center square, 0 elsewhere
    center_x, center_y = 0.5, 0.5
    square_size = 0.2  # Larger square for easier learning
    in_square = ((X_flat >= center_x - square_size/2) & (X_flat <= center_x + square_size/2) & 
                (Y_flat >= center_y - square_size/2) & (Y_flat <= center_y + square_size/2))
    
    u_target = torch.ones_like(X_flat)  # Constant u=1
    v_target = torch.where(in_square, torch.tensor(0.25), torch.tensor(0.0))  # Step function for v
    
    print(f"Step function stats:")
    print(f"  U: min={u_target.min():.6f}, max={u_target.max():.6f}, range={u_target.max()-u_target.min():.6f}")
    print(f"  V: min={v_target.min():.6f}, max={v_target.max():.6f}, range={v_target.max()-v_target.min():.6f}")
    print(f"  Square fraction: {in_square.float().mean():.3f}")
    
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
        
        # Check how well it learned the step
        v_in_square = v_pred[in_square.squeeze()]
        v_out_square = v_pred[~in_square.squeeze()]
        print(f"  V inside square: mean={v_in_square.mean():.6f}, std={v_in_square.std():.6f} (target: 0.25)")
        print(f"  V outside square: mean={v_out_square.mean():.6f}, std={v_out_square.std():.6f} (target: 0.0)")
        
        # Correlation
        u_corr = torch.corrcoef(torch.stack([u_pred.squeeze(), u_target.squeeze()]))[0,1]
        v_corr = torch.corrcoef(torch.stack([v_pred.squeeze(), v_target.squeeze()]))[0,1]
        print(f"  Correlation: U={u_corr:.6f}, V={v_corr:.6f}")
    
    return v_corr.item()

def test_sine_baseline():
    """Baseline test - sin function that we know works"""
    print("\n🔍 BASELINE: SINE FUNCTION TEST")
    print("=" * 50)
    
    n_per_dim = 64
    x = torch.linspace(0, 1, n_per_dim).reshape(-1, 1)
    y = torch.linspace(0, 1, n_per_dim).reshape(-1, 1)
    X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    t = torch.zeros_like(X_flat)
    F = torch.full_like(X_flat, 0.055)
    k = torch.full_like(X_flat, 0.062)

    # Target function
    u_target = torch.sin(np.pi * X_flat) * torch.sin(np.pi * Y_flat)
    v_target = torch.zeros_like(u_target)

    # Model
    model = GrayScottPINN(hidden_size=50, num_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        u_pred, v_pred = model(X_flat, Y_flat, t, F, k)
        loss = torch.mean((u_pred - u_target) ** 2) + torch.mean((v_pred - v_target) ** 2)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.8f}")

    with torch.no_grad():
        u_pred, v_pred = model(X_flat, Y_flat, t, F, k)
        u_corr = torch.corrcoef(torch.stack([u_pred.squeeze(), u_target.squeeze()]))[0,1]
        print(f"Final sine correlation: {u_corr:.6f}")
    
    return u_corr.item()

if __name__ == "__main__":
    print("🧠 Gray-Scott PINN IC Learning Diagnosis")
    print("=" * 60)
    
    # Run all tests
    sine_corr = test_sine_baseline()
    step_corr = test_step_function_learning()
    ic_u_corr, ic_v_corr = test_ic_only_training()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Sine function correlation:     {sine_corr:.6f} {'(PERFECT)' if sine_corr > 0.99 else '(POOR)'}")
    print(f"🔳 Step function correlation:     {step_corr:.6f} {'(GOOD)' if step_corr > 0.8 else '(POOR)'}")
    print(f"🎯 IC U correlation:              {ic_u_corr:.6f} {'(GOOD)' if ic_u_corr > 0.8 else '(POOR)'}")
    print(f"🎯 IC V correlation:              {ic_v_corr:.6f} {'(GOOD)' if ic_v_corr > 0.8 else '(POOR)'}")
    
    print("\n🔍 DIAGNOSIS:")
    if sine_corr > 0.99 and step_corr < 0.5:
        print("❌ Model CANNOT learn step functions - fundamental limitation!")
        print("   → Neural networks struggle with discontinuous functions")
        print("   → Need different approach: smooth IC or different architecture")
    elif sine_corr > 0.99 and ic_u_corr > 0.8 and ic_v_corr > 0.8:
        print("✅ IC learning works when isolated!")
        print("   → Problem is interaction with physics/boundary losses")
        print("   → Solution: Better loss balancing or sequential training")
    elif sine_corr < 0.99:
        print("❌ Model has fundamental problems even with smooth functions")
        print("   → Check model architecture, learning rate, etc.")
    else:
        print("🤔 Mixed results - need further investigation") 