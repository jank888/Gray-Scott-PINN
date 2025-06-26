import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import time

class GrayScott1DPINN(nn.Module):
    def __init__(self, hidden_size=20, num_layers=4):
        """
        1D Gray-Scott PINN Model
        Architecture: 4 hidden layers with 20 neurons each (as per article)
        Activation: Hyperbolic tangent (tanh)
        """
        super(GrayScott1DPINN, self).__init__()
        
        # Network architecture exactly as specified in the article
        layers = []
        
        # Input layer: [x, t] -> 2 dimensions
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())
        
        # Hidden layers: 4 layers with 20 neurons each
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        # Output layer: [u, v] -> 2 dimensions
        layers.append(nn.Linear(hidden_size, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Gray-Scott parameters from the article
        self.Du = 0.01  # Diffusion coefficient for u
        self.Dv = 0.01  # Diffusion coefficient for v
        self.F = 0.09   # Feed rate
        self.K = -0.004 # Kill rate (note: negative as per article)
        
        # Domain parameters
        self.L = 2.0    # Domain length
        self.us = 1.0   # Steady state value for u
        self.vs = 0.0   # Steady state value for v
    
    def forward(self, x, t):
        """Forward pass through the network"""
        # Normalize inputs for better training stability
        x_norm = x / self.L  # Normalize x from [0, L] to [0, 1]
        t_norm = t / 15.0    # Normalize t to reasonable range
        
        inputs = torch.cat([x_norm, t_norm], dim=1)
        output = self.network(inputs)
        
        u = output[:, 0:1]
        v = output[:, 1:2]
        
        return u, v
    
    def pde_residual(self, x, t):
        """
        Compute PDE residuals for the 1D Gray-Scott system:
        ∂u/∂t = Du * ∂²u/∂x² - uv² + F(1-u)
        ∂v/∂t = Dv * ∂²v/∂x² + uv² - (F+K)v
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v = self.forward(x, t)
        
        # First-order derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                                  create_graph=True, retain_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
        
        # Second-order derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True, retain_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                                   create_graph=True, retain_graph=True)[0]
        
        # Gray-Scott equations
        pde_u = u_t - (self.Du * u_xx - u * v**2 + self.F * (1 - u))
        pde_v = v_t - (self.Dv * v_xx + u * v**2 - (self.F + self.K) * v)
        
        return pde_u, pde_v
    
    def initial_condition_loss(self, x):
        """
        Initial conditions from the article:
        u(x, 0) = us + 0.01 * sin(πx/L)
        v(x, 0) = vs - 0.12 * sin(πx/L)
        """
        t_zero = torch.zeros_like(x)
        u_pred, v_pred = self.forward(x, t_zero)
        
        # Target initial conditions
        u_target = self.us + 0.01 * torch.sin(np.pi * x / self.L)
        v_target = self.vs - 0.12 * torch.sin(np.pi * x / self.L)
        
        ic_loss_u = torch.mean((u_pred - u_target) ** 2)
        ic_loss_v = torch.mean((v_pred - v_target) ** 2)
        
        return ic_loss_u + ic_loss_v
    
    def boundary_condition_loss(self, t):
        """
        Boundary conditions from the article:
        u(0, t) = u(L, t) = us
        v(0, t) = v(L, t) = vs
        """
        x_left = torch.zeros_like(t)
        x_right = torch.full_like(t, self.L)
        
        u_left, v_left = self.forward(x_left, t)
        u_right, v_right = self.forward(x_right, t)
        
        # Target boundary values
        u_target = torch.full_like(t, self.us)
        v_target = torch.full_like(t, self.vs)
        
        bc_loss = (torch.mean((u_left - u_target)**2) + torch.mean((v_left - v_target)**2) +
                   torch.mean((u_right - u_target)**2) + torch.mean((v_right - v_target)**2))
        
        return bc_loss

def generate_training_points(N0=100, Nb=200, Nr=1024, L=2.0, t_max=15.0):
    """
    Generate training points as specified in the article:
    N0 = 100 for initial conditions
    Nb = 200 for boundary conditions  
    Nr = 1024 for collocation points
    Total: N = N0 + Nb + Nr = 1324 points
    """
    
    # Initial condition points (N0 = 100)
    x_ic = torch.linspace(0, L, N0).reshape(-1, 1)
    t_ic = torch.zeros(N0, 1)
    
    # Boundary condition points (Nb = 200)
    t_bc = torch.rand(Nb, 1) * t_max  # Random time points
    x_bc = torch.cat([torch.zeros(Nb//2, 1), torch.full((Nb//2, 1), L)])  # x=0 and x=L
    
    # Collocation points (Nr = 1024) 
    x_col = torch.rand(Nr, 1) * L     # Random x in [0, L]
    t_col = torch.rand(Nr, 1) * t_max # Random t in [0, t_max]
    
    return (x_ic, t_ic), (x_bc, t_bc), (x_col, t_col)

def train_pinn(model, epochs=20000, lr=1e-3):
    """
    Train the 1D Gray-Scott PINN model
    Uses Adam optimizer with learning rate 10^-3 as specified in the article
    """
    print("Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    # Generate training points
    print("Generating training points...")
    (x_ic, t_ic), (x_bc, t_bc), (x_col, t_col) = generate_training_points()
    
    print(f"Training points: {len(x_ic)} initial + {len(x_bc)} boundary + {len(x_col)} collocation = {len(x_ic) + len(x_bc) + len(x_col)} total")
    print("Starting training...")
    
    losses = []
    train_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        optimizer.zero_grad()
        
        # Physics loss (PDE residuals at collocation points)
        pde_u, pde_v = model.pde_residual(x_col, t_col)
        physics_loss = torch.mean(pde_u**2) + torch.mean(pde_v**2)
        
        # Initial condition loss
        ic_loss = model.initial_condition_loss(x_ic)
        
        # Boundary condition loss
        bc_loss = model.boundary_condition_loss(t_bc)
        
        # Total loss (weighted combination)
        total_loss = physics_loss + 10 * ic_loss + 10 * bc_loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(total_loss.item())
        epoch_time = time.time() - epoch_start_time
        
        # Progress reporting
        if epoch == 0:
            print(f'Epoch {epoch} completed in {epoch_time:.3f}s, Loss: {total_loss.item():.6f}')
        elif epoch < 100 and epoch % 10 == 0:
            elapsed = time.time() - train_start_time
            avg_time_per_epoch = elapsed / (epoch + 1)
            eta = avg_time_per_epoch * (epochs - epoch - 1)
            print(f'Epoch {epoch}, Loss: {total_loss.item():.6f}, Physics: {physics_loss.item():.6f}, IC: {ic_loss.item():.6f}, BC: {bc_loss.item():.6f}')
            print(f'  Time/epoch: {epoch_time:.3f}s, ETA: {eta/60:.1f} minutes')
        elif epoch % 1000 == 0:
            elapsed = time.time() - train_start_time
            avg_time_per_epoch = elapsed / (epoch + 1)
            eta = avg_time_per_epoch * (epochs - epoch - 1)
            print(f'Epoch {epoch}, Loss: {total_loss.item():.6f}, Physics: {physics_loss.item():.6f}, IC: {ic_loss.item():.6f}, BC: {bc_loss.item():.6f}')
            print(f'  Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min, Avg: {avg_time_per_epoch:.3f}s/epoch')
    
    total_time = time.time() - train_start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    return losses

def visualize_solution(model, L=2.0, t_max=15.0, nx=200, nt=100):
    """Visualize the 1D Gray-Scott solution"""
    x = torch.linspace(0, L, nx).reshape(-1, 1)
    t = torch.linspace(0, t_max, nt).reshape(-1, 1)
    
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)
    
    with torch.no_grad():
        u_pred, v_pred = model(X_flat, T_flat)
        U = u_pred.reshape(nx, nt).numpy()
        V = v_pred.reshape(nx, nt).numpy()
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # u(x,t) solution
    im1 = ax1.contourf(T.numpy(), X.numpy(), U, levels=50, cmap='viridis')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Position x')
    ax1.set_title('u(x,t) - Component u')
    plt.colorbar(im1, ax=ax1)
    
    # v(x,t) solution
    im2 = ax2.contourf(T.numpy(), X.numpy(), V, levels=50, cmap='plasma')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Position x')
    ax2.set_title('v(x,t) - Component v')
    plt.colorbar(im2, ax=ax2)
    
    # u at different times
    times = [0, 3, 6, 9, 12, 15]
    x_plot = torch.linspace(0, L, 200).reshape(-1, 1)
    for i, t_val in enumerate(times):
        t_plot = torch.full_like(x_plot, t_val)
        with torch.no_grad():
            u_plot, _ = model(x_plot, t_plot)
        ax3.plot(x_plot.numpy(), u_plot.numpy(), label=f't={t_val}')
    ax3.set_xlabel('Position x')
    ax3.set_ylabel('u(x,t)')
    ax3.set_title('u profiles at different times')
    ax3.legend()
    ax3.grid(True)
    
    # v at different times
    for i, t_val in enumerate(times):
        t_plot = torch.full_like(x_plot, t_val)
        with torch.no_grad():
            _, v_plot = model(x_plot, t_plot)
        ax4.plot(x_plot.numpy(), v_plot.numpy(), label=f't={t_val}')
    ax4.set_xlabel('Position x')
    ax4.set_ylabel('v(x,t)')
    ax4.set_title('v profiles at different times')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    """Main training and visualization function"""
    print("=" * 60)
    print("1D Gray-Scott PINN Training")
    print("Article Case Study Implementation")
    print("=" * 60)
    print()
    print("Parameters:")
    print("- Du = Dv = 0.01")
    print("- F = 0.09, K = -0.004")
    print("- Architecture: 4 hidden layers, 20 neurons each")
    print("- Activation: tanh")
    print("- Training points: N0=100, Nb=200, Nr=1024")
    print("- Domain: L = 2")
    print()
    
    # Create model with exact specifications from article
    model = GrayScott1DPINN(hidden_size=20, num_layers=4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Confirm before starting
    confirm = input("Ready to start training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print()
    
    # Train the model
    losses = train_pinn(model, epochs=20000, lr=1e-3)
    
    # Save the trained model
    model_name = '1d_gray_scott_model.pth'
    torch.save(model.state_dict(), model_name)
    print(f"✅ Model saved as '{model_name}'")
    
    # Save model metadata
    metadata = {
        'hidden_size': 20,
        'num_layers': 4,
        'activation': 'tanh',
        'Du': 0.01,
        'Dv': 0.01,
        'F': 0.09,
        'K': -0.004,
        'L': 2.0,
        'us': 1.0,
        'vs': 0.0,
        'epochs': 20000,
        'lr': 1e-3,
        'N0': 100,
        'Nb': 200,
        'Nr': 1024,
        'description': 'Article case study implementation'
    }
    metadata_name = '1d_gray_scott_model_metadata.json'
    with open(metadata_name, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Model metadata saved as '{metadata_name}'")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('1D Gray-Scott PINN Training Loss')
    plt.yscale('log')
    plt.grid(True)
    loss_plot_name = '1d_gray_scott_training_loss.png'
    plt.savefig(loss_plot_name)
    print(f"✅ Training loss plot saved as '{loss_plot_name}'")
    
    # Generate visualization
    print("\nGenerating solution visualization...")
    fig = visualize_solution(model)
    viz_name = '1d_gray_scott_solution.png'
    fig.savefig(viz_name, dpi=300, bbox_inches='tight')
    print(f"✅ Solution visualization saved as '{viz_name}'")
    
    print()
    print("🎉 Training completed successfully!")
    print("Files generated:")
    print(f"- {model_name}")
    print(f"- {metadata_name}")
    print(f"- {loss_plot_name}")
    print(f"- {viz_name}")
    
    # Ask if user wants to show the plots
    show_plots = input("\nShow plots? (y/n): ").strip().lower()
    if show_plots == 'y':
        plt.show()
    else:
        plt.close('all')

if __name__ == "__main__":
    main()
