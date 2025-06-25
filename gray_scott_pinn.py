import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

class GrayScottPINN(nn.Module):
    def __init__(self, hidden_size=100, num_layers=5):
        super(GrayScottPINN, self).__init__()
        
        # Input: [x, y, t, F, k] -> 5 dimensions
        # Output: [u, v] -> 2 dimensions
        
        layers = []
        layers.append(nn.Linear(5, hidden_size))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_size, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Gray-Scott parameters
        self.Du = 0.16  # Diffusion coefficient for u
        self.Dv = 0.08  # Diffusion coefficient for v
    
    def forward(self, x, y, t, F, k):
        # Normalize inputs to similar ranges for better training
        # x, y already in [0, 1]
        t_norm = t / 15.0  # Normalize t from [0, 15] to [0, 1]
        F_norm = (F - 0.01) / 0.1  # Normalize F from [0.01, 0.11] to [0, 1]  
        k_norm = (k - 0.01) / 0.1  # Normalize k from [0.01, 0.11] to [0, 1]
        
        inputs = torch.cat([x, y, t_norm, F_norm, k_norm], dim=1)
        output = self.network(inputs)
        u = output[:, 0:1]  # Remove sigmoid - let physics constrain the range
        v = output[:, 1:2]  # Remove sigmoid - let physics constrain the range
        return u, v
    
    def pde_residual(self, x, y, t, F, k):
        x.requires_grad_(True)
        y.requires_grad_(True) 
        t.requires_grad_(True)
        
        u, v = self.forward(x, y, t, F, k)
        
        # Compute gradients
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                                  create_graph=True, retain_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True, retain_graph=True)[0]
        
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                   create_graph=True, retain_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                                   create_graph=True, retain_graph=True)[0]
        
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y),
                                   create_graph=True, retain_graph=True)[0]
        
        # Laplacians
        laplacian_u = u_xx + u_yy
        laplacian_v = v_xx + v_yy
        
        # Gray-Scott equations
        # du/dt = Du * ∇²u - uv² + F(1-u)
        # dv/dt = Dv * ∇²v + uv² - (F+k)v
        
        pde_u = u_t - (self.Du * laplacian_u - u * v**2 + F * (1 - u))
        pde_v = v_t - (self.Dv * laplacian_v + u * v**2 - (F + k) * v)
        
        return pde_u, pde_v
    
    def initial_condition_loss(self, x, y, F, k):
        t_zero = torch.zeros_like(x)
        u, v = self.forward(x, y, t_zero, F, k)

        # SMOOTH Gray-Scott initial condition with 3x3 grid of bumps:
        # Multiple perturbations create richer pattern formation dynamics
        
        # 3x3 grid of bump centers
        bump_centers = [
            (0.25, 0.25), (0.5, 0.25), (0.75, 0.25),  # Bottom row
            (0.25, 0.5),  (0.5, 0.5),  (0.75, 0.5),   # Middle row  
            (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)   # Top row
        ]
        
        radius = 0.06  # Smaller radius since we have multiple bumps
        sigmoid_sharpness = 20.0
        
        # Initialize as base values
        u_target = torch.ones_like(x)  # Base value u = 1
        v_target = torch.zeros_like(x)  # Base value v = 0
        
        # Add each bump to the initial condition
        for center_x, center_y in bump_centers:
            # Distance from this bump center
            dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            # Smooth bump: u decreases, v increases near center
            u_bump = -0.5 * torch.sigmoid(sigmoid_sharpness * (radius - dist))
            v_bump = 0.25 * torch.sigmoid(sigmoid_sharpness * (radius - dist))
            
            # Add this bump's contribution
            u_target = u_target + u_bump
            v_target = v_target + v_bump
        
        # Clamp to reasonable ranges to avoid overlap issues
        u_target = torch.clamp(u_target, 0.4, 1.0)
        v_target = torch.clamp(v_target, 0.0, 0.3)

        ic_loss_u = torch.mean((u - u_target) ** 2)
        ic_loss_v = torch.mean((v - v_target) ** 2)
        return ic_loss_u + ic_loss_v
    
    def boundary_condition_loss(self, x_bound, y_bound, t_bound, F_bound, k_bound):
        """
        Periodic boundary conditions: u(0,y,t) = u(1,y,t) and u(x,0,t) = u(x,1,t)
        More physically realistic for pattern formation in infinite domains
        """
        # Left and right boundaries (x=0 and x=1)
        x_left = torch.zeros_like(y_bound)
        x_right = torch.ones_like(y_bound)
        
        u_left, v_left = self.forward(x_left, y_bound, t_bound, F_bound, k_bound)
        u_right, v_right = self.forward(x_right, y_bound, t_bound, F_bound, k_bound)
        
        # Bottom and top boundaries (y=0 and y=1)
        y_bottom = torch.zeros_like(x_bound)
        y_top = torch.ones_like(x_bound)
        
        u_bottom, v_bottom = self.forward(x_bound, y_bottom, t_bound, F_bound, k_bound)
        u_top, v_top = self.forward(x_bound, y_top, t_bound, F_bound, k_bound)
        
        # Periodic boundary conditions: opposite sides must be equal
        bc_loss = (torch.mean((u_left - u_right)**2) + torch.mean((v_left - v_right)**2) +
                  torch.mean((u_bottom - u_top)**2) + torch.mean((v_bottom - v_top)**2))
        
        return bc_loss

def generate_dense_time_samples(n_samples, t_max=15.0, decay_rate=2.5):
    """
    Generate time samples with exponentially decreasing density from t=0
    More natural distribution for physics learning
    
    Args:
        n_samples: Total number of time samples
        t_max: Maximum time value
        decay_rate: Controls how quickly density decreases (higher = more concentrated near t=0)
    
    Returns:
        torch.Tensor: Time samples with shape (n_samples, 1)
    """
    # Generate uniform random samples
    uniform_samples = torch.rand(n_samples, 1)
    
    # Transform using exponential distribution
    # This creates natural exponential decay from t=0
    t_samples = -torch.log(1 - uniform_samples * (1 - torch.exp(torch.tensor(-decay_rate)))) / decay_rate * t_max
    
    # Clamp to ensure we stay within [0, t_max]
    t_samples = torch.clamp(t_samples, 0, t_max)
    
    return t_samples

def generate_boundary_points(n_points=500):
    """Generate points on the domain boundaries for periodic boundary conditions"""
    # For periodic BCs, we need points on opposite boundaries to match
    # Generate pairs of points: (0,y) and (1,y), (x,0) and (x,1)
    
    # Horizontal boundaries: y=0 and y=1 with same x coordinates
    x_horizontal = torch.rand(n_points//2, 1)  # Same x for bottom and top
    y_bottom = torch.zeros(n_points//2, 1)
    y_top = torch.ones(n_points//2, 1)
    
    # Vertical boundaries: x=0 and x=1 with same y coordinates  
    y_vertical = torch.rand(n_points//2, 1)  # Same y for left and right
    x_left = torch.zeros(n_points//2, 1)
    x_right = torch.ones(n_points//2, 1)
    
    # Combine all boundary points
    x_bound = torch.cat([x_horizontal, x_horizontal, x_left, x_right], dim=0)
    y_bound = torch.cat([y_bottom, y_top, y_vertical, y_vertical], dim=0)
    
    # Generate time and parameter values for boundary points
    t_bound = generate_dense_time_samples(len(x_bound))  # Dense near t=0
    F_bound = torch.rand(len(x_bound), 1) * 0.1 + 0.01  # [0.01, 0.11]
    k_bound = torch.rand(len(x_bound), 1) * 0.1 + 0.01  # [0.01, 0.11]
    
    return x_bound, y_bound, t_bound, F_bound, k_bound

def generate_training_points(n_points=10000):
    """Generate training points using uniform grid sampling for x and y."""
    n_per_dim = int(np.sqrt(n_points))
    x = torch.linspace(0, 1, n_per_dim)
    y = torch.linspace(0, 1, n_per_dim)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    # Sample t with dense distribution near 0, F, k randomly for each spatial point
    t = generate_dense_time_samples(len(X))  # Dense sampling near t=0
    F = torch.rand_like(X) * 0.1 + 0.01  # [0.01, 0.11]
    k = torch.rand_like(X) * 0.1 + 0.01  # [0.01, 0.11]
    return X, Y, t, F, k

def generate_ic_grid_points(n_per_dim=32):
    """Generate a uniform grid of IC points over the (x, y) domain."""
    x = torch.linspace(0, 1, n_per_dim)
    y = torch.linspace(0, 1, n_per_dim)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    t = torch.zeros_like(X)
    F = torch.full_like(X, 0.055)  # or random in [0.01, 0.11]
    k = torch.full_like(X, 0.062)  # or random in [0.01, 0.11]
    return X, Y, t, F, k

def train_pinn(model, epochs=20000, lr=1e-3):
    """Train the PINN model"""
    import time
    
    print("Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    # Generate training points ONCE (not every epoch!)
    print("Generating training points...")
    start_time = time.time()
    
    # Generate uniform spatial grid
    n_per_dim = int(np.sqrt(2000))  # ~45x45 = 2025 points
    x_grid = torch.linspace(0, 1, n_per_dim)
    y_grid = torch.linspace(0, 1, n_per_dim)
    X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
    X_spatial = X_grid.reshape(-1, 1)
    Y_spatial = Y_grid.reshape(-1, 1)
    
    # Physics points: same spatial locations, dense t near 0, random F, k
    x_train, y_train = X_spatial, Y_spatial
    t_train = generate_dense_time_samples(len(X_spatial))  # Dense sampling near t=0
    F_train = torch.rand_like(X_spatial) * 0.1 + 0.01  # [0.01, 0.11]
    k_train = torch.rand_like(X_spatial) * 0.1 + 0.01  # [0.01, 0.11]
    
    # IC points: same spatial locations, t=0, fixed F, k
    x_ic, y_ic = X_spatial, Y_spatial
    t_ic = torch.zeros_like(X_spatial)
    F_ic = torch.full_like(X_spatial, 0.055)
    k_ic = torch.full_like(X_spatial, 0.062)
    
    x_bound, y_bound, t_bound, F_bound, k_bound = generate_boundary_points(400)  # Boundary points
    
    setup_time = time.time() - start_time
    print(f"Setup completed in {setup_time:.2f} seconds")
    print(f"Training points: {len(x_train)} physics (dense t) + {len(x_ic)} initial + {len(x_bound)} boundary (dense t)")
    print("Starting training...")
    
    losses = []
    train_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        optimizer.zero_grad()
        
        # Physics loss
        pde_u, pde_v = model.pde_residual(x_train, y_train, t_train, F_train, k_train)
        physics_loss = torch.mean(pde_u**2) + torch.mean(pde_v**2)
        
        # Initial condition loss - computed like the working sine test
        u_ic_pred, v_ic_pred = model(x_ic, y_ic, t_ic, F_ic, k_ic)
        
        # Create SMOOTH IC targets with 3x3 grid (same logic as in initial_condition_loss method)
        bump_centers = [
            (0.25, 0.25), (0.5, 0.25), (0.75, 0.25),  # Bottom row
            (0.25, 0.5),  (0.5, 0.5),  (0.75, 0.5),   # Middle row  
            (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)   # Top row
        ]
        
        radius = 0.06
        sigmoid_sharpness = 20.0
        
        # Initialize as base values
        u_target_ic = torch.ones_like(x_ic)  # Base value u = 1
        v_target_ic = torch.zeros_like(x_ic)  # Base value v = 0
        
        # Add each bump to the initial condition
        for center_x, center_y in bump_centers:
            # Distance from this bump center
            dist = torch.sqrt((x_ic - center_x) ** 2 + (y_ic - center_y) ** 2)
            
            # Smooth bump: u decreases, v increases near center
            u_bump = -0.5 * torch.sigmoid(sigmoid_sharpness * (radius - dist))
            v_bump = 0.25 * torch.sigmoid(sigmoid_sharpness * (radius - dist))
            
            # Add this bump's contribution
            u_target_ic = u_target_ic + u_bump
            v_target_ic = v_target_ic + v_bump
        
        # Clamp to reasonable ranges
        u_target_ic = torch.clamp(u_target_ic, 0.4, 1.0)
        v_target_ic = torch.clamp(v_target_ic, 0.0, 0.3)
        
        ic_loss = torch.mean((u_ic_pred - u_target_ic) ** 2) + torch.mean((v_ic_pred - v_target_ic) ** 2)
        
        # Boundary condition loss
        bc_loss = model.boundary_condition_loss(x_bound, y_bound, t_bound, F_bound, k_bound)
        
        # Total loss (weighted combination)
        total_loss = 2 * ic_loss + physics_loss + bc_loss  # Heavy IC weight to ensure it's learned first
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(total_loss.item())
        epoch_time = time.time() - epoch_start_time
        
        # More frequent updates, especially early on
        if epoch == 0:
            print(f'Epoch {epoch} completed in {epoch_time:.3f}s, Loss: {total_loss.item():.6f}')
        elif epoch < 100 and epoch % 10 == 0:
            elapsed = time.time() - train_start_time
            avg_time_per_epoch = elapsed / (epoch + 1)
            eta = avg_time_per_epoch * (epochs - epoch - 1)
            print(f'Epoch {epoch}, Loss: {total_loss.item():.6f}, Physics: {physics_loss.item():.6f}, IC: {ic_loss.item():.6f}, BC: {bc_loss.item():.6f}')
            print(f'  Time/epoch: {epoch_time:.3f}s, ETA: {eta/60:.1f} minutes')
        elif epoch % 500 == 0:
            elapsed = time.time() - train_start_time
            avg_time_per_epoch = elapsed / (epoch + 1)
            eta = avg_time_per_epoch * (epochs - epoch - 1)
            print(f'Epoch {epoch}, Loss: {total_loss.item():.6f}, Physics: {physics_loss.item():.6f}, IC: {ic_loss.item():.6f}, BC: {bc_loss.item():.6f}')
            print(f'  Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min, Avg: {avg_time_per_epoch:.3f}s/epoch')
    
    total_time = time.time() - train_start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    return losses



if __name__ == "__main__":


    print("=" * 60)
    print("Gray-Scott PINN Training")
    print("=" * 60)
    print()
    print("Choose training mode:")
    print("1. Fast Training (5-10 minutes)")
    print("   - Smaller model for quick testing")
    print("   - 5,000 epochs, 50 neurons, 4 layers")
    print()
    print("2. Standard Training (15-30 minutes)")
    print("   - Full model for best results")
    print("   - 15,000 epochs, 100 neurons, 5 layers")
    print()
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    print()
    
    if choice == '1':
        print(" FAST TRAINING MODE SELECTED")
        print("This will train a smaller model quickly for testing")
        fast_mode = True
        epochs = 10000
        model = GrayScottPINN(hidden_size=50, num_layers=4)  # Smaller model
    else:
        print("STANDARD TRAINING MODE SELECTED")
        print("Training full model for best results")
        fast_mode = False
        epochs = 30000
        model = GrayScottPINN()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Confirm before starting
    confirm = input("Ready to start training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        exit()
    
    print()
    losses = train_pinn(model, epochs=epochs)
    
    # Save the trained model
    model_name = 'gray_scott_pinn_model_fast.pth' if fast_mode else 'gray_scott_pinn_model.pth'
    torch.save(model.state_dict(), model_name)
    print(f" Model saved as '{model_name}'")
    
    # Save model metadata for easier loading
    metadata = {
        'hidden_size': 50 if fast_mode else 100,
        'num_layers': 4 if fast_mode else 5,
        'epochs': epochs,
        'mode': 'fast' if fast_mode else 'standard'
    }
    metadata_name = model_name.replace('.pth', '_metadata.json')
    import json
    with open(metadata_name, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f" Model metadata saved as '{metadata_name}'")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'PINN Training Loss ({epochs} epochs)')
    plt.yscale('log')
    plt.grid(True)
    loss_plot_name = 'training_loss_fast.png' if fast_mode else 'training_loss.png'
    plt.savefig(loss_plot_name)
    print(f" Training loss plot saved as '{loss_plot_name}'")
    
    print()
    print(" Training completed successfully!")
    print("You can now run the interactive visualizer:")
    print("   streamlit run app.py")
    
    # Ask if user wants to show the plot
    show_plot = input("\nShow training loss plot? (y/n): ").strip().lower()
    if show_plot == 'y':
        plt.show()
    else:
        plt.close()
    
    # Print some weights and biases
    for name, param in model.named_parameters():
        print(f"{name}: min={param.min().item():.4f}, max={param.max().item():.4f}")
    
    
    