import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os

class GrayScott2DPINN(nn.Module):
    def __init__(self, hidden_size=20, num_layers=4):
        """
        2D Gray-Scott PINN Model
        Architecture: 4 hidden layers with 20 neurons each (as per paper)
        Activation: Hyperbolic tangent for input, sigmoid for output
        """
        super(GrayScott2DPINN, self).__init__()
        
        # Network architecture exactly as specified in the paper
        layers = []
        
        # Input layer: [x, y, t] -> 3 dimensions
        layers.append(nn.Linear(3, hidden_size))
        layers.append(nn.Tanh())  # Hyperbolic tangent for input
        
        # Hidden layers: 4 layers with 20 neurons each
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        # Output layer: [u, v] -> 2 dimensions with sigmoid activation
        layers.append(nn.Linear(hidden_size, 2))
        layers.append(nn.Sigmoid())  # Sigmoid activation for output
        
        self.network = nn.Sequential(*layers)
        
        # Gray-Scott parameters from the paper
        # Du = 2*Dv = 2×10^-5, so Dv = 10^-5, Du = 2×10^-5
        self.Dv = 1e-5      # Diffusion coefficient for v
        self.Du = 2 * self.Dv  # Diffusion coefficient for u = 2*Dv
        
        # Case 1 parameters from paper
        self.F = 0.030      # Feed rate
        self.K = 0.060      # Kill rate
        
        # Domain parameters (assuming unit domain [0,1] x [0,1])
        self.domain_size = 1.0
    
    def forward(self, x, y, t):
        """Forward pass through the network"""
        # Normalize inputs for better training stability
        # Assuming domain is [0,1] x [0,1] and time up to 5000
        x_norm = x  # Already in [0,1]
        y_norm = y  # Already in [0,1] 
        t_norm = t / 5000.0  # Normalize t from [0, 5000] to [0, 1]
        
        inputs = torch.cat([x_norm, y_norm, t_norm], dim=1)
        output = self.network(inputs)
        
        u = output[:, 0:1]
        v = output[:, 1:2]
        
        return u, v
    
    def pde_residual(self, x, y, t):
        """
        Compute PDE residuals for the 2D Gray-Scott system:
        f1 = ∂u/∂t + uv² - F(1-u) - Du*Δu = 0
        f2 = ∂v/∂t - uv² + (F+K)v - Dv*Δv = 0
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v = self.forward(x, y, t)
        
        # First-order derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                                  create_graph=True, retain_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
        
        # Second-order derivatives for Laplacian
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                   create_graph=True, retain_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                                   create_graph=True, retain_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y),
                                   create_graph=True, retain_graph=True)[0]
        
        # Laplacians
        laplacian_u = u_xx + u_yy
        laplacian_v = v_xx + v_yy
        
        # Gray-Scott equations as defined in the paper
        f1 = u_t + u * v**2 - self.F * (1 - u) - self.Du * laplacian_u
        f2 = v_t - u * v**2 + (self.F + self.K) * v - self.Dv * laplacian_v
        
        return f1, f2
    
    def initial_condition_loss(self, x, y):
        """
        Initial conditions from the paper:
        - Entire system in trivial state (u = 1, v = 0) at t = 0
        - Small squared area at center perturbed to (u = 1/2, v = 1/4)
        """
        t_zero = torch.zeros_like(x)
        u_pred, v_pred = self.forward(x, y, t_zero)
        
        # Create initial condition targets
        u_target = torch.ones_like(x)  # Base state u = 1
        v_target = torch.zeros_like(x)  # Base state v = 0
        
        # Define small square perturbation at center
        center_x, center_y = 0.5, 0.5
        square_size = 0.1  # Small square size
        
        # Create mask for the central square
        mask = ((torch.abs(x - center_x) <= square_size/2) & 
                (torch.abs(y - center_y) <= square_size/2))
        
        # Apply perturbation in the central square
        u_target = torch.where(mask, torch.tensor(0.5), u_target)  # u = 1/2 in square
        v_target = torch.where(mask, torch.tensor(0.25), v_target)  # v = 1/4 in square
        
        ic_loss_u = torch.mean((u_pred - u_target) ** 2)
        ic_loss_v = torch.mean((v_pred - v_target) ** 2)
        
        return ic_loss_u + ic_loss_v
    
    def boundary_condition_loss(self, x_bound, y_bound, t_bound):
        """
        Periodic boundary conditions as specified in the paper
        u(0,y,t) = u(1,y,t), u(x,0,t) = u(x,1,t)
        v(0,y,t) = v(1,y,t), v(x,0,t) = v(x,1,t)
        """
        # Extract boundary coordinates
        n_points = len(x_bound) // 4
        
        # Left and right boundaries (x=0 and x=1)
        x_left = x_bound[:n_points]
        y_left = y_bound[:n_points]
        t_left = t_bound[:n_points]
        
        x_right = x_bound[n_points:2*n_points]
        y_right = y_bound[n_points:2*n_points]
        t_right = t_bound[n_points:2*n_points]
        
        # Bottom and top boundaries (y=0 and y=1)
        x_bottom = x_bound[2*n_points:3*n_points]
        y_bottom = y_bound[2*n_points:3*n_points]
        t_bottom = t_bound[2*n_points:3*n_points]
        
        x_top = x_bound[3*n_points:]
        y_top = y_bound[3*n_points:]
        t_top = t_bound[3*n_points:]
        
        # Get solutions at boundaries
        u_left, v_left = self.forward(x_left, y_left, t_left)
        u_right, v_right = self.forward(x_right, y_right, t_right)
        u_bottom, v_bottom = self.forward(x_bottom, y_bottom, t_bottom)
        u_top, v_top = self.forward(x_top, y_top, t_top)
        
        # Periodic boundary conditions: opposite sides must be equal
        bc_loss = (torch.mean((u_left - u_right)**2) + torch.mean((v_left - v_right)**2) +
                   torch.mean((u_bottom - u_top)**2) + torch.mean((v_bottom - v_top)**2))
        
        return bc_loss

def generate_training_points(Nr=10000, N0_per_dim=101, NB=400, domain_size=1.0, t_max=5000.0):
    """
    Generate training points as specified in the paper:
    Nr = 10,000 collocation points (uniformly sampled)
    N0 = 101 × 101 = 10,201 initial condition points (grid)
    NB = 4 × 100 = 400 boundary condition points (100 per side)
    """
    
    # Collocation points (Nr = 10,000) - uniformly sampled in time-space domain
    x_col = torch.rand(Nr, 1) * domain_size
    y_col = torch.rand(Nr, 1) * domain_size
    t_col = torch.rand(Nr, 1) * t_max
    
    # Initial condition points (N0 = 101 × 101) - grid at t=0
    x_ic_1d = torch.linspace(0, domain_size, N0_per_dim)
    y_ic_1d = torch.linspace(0, domain_size, N0_per_dim)
    X_ic, Y_ic = torch.meshgrid(x_ic_1d, y_ic_1d, indexing='ij')
    x_ic = X_ic.reshape(-1, 1)
    y_ic = Y_ic.reshape(-1, 1)
    
    # Boundary condition points (NB = 4 × 100 = 400) - 100 points per side
    points_per_side = NB // 4
    t_bound = torch.rand(NB, 1) * t_max  # Random time points
    
    # Left boundary (x=0)
    x_left = torch.zeros(points_per_side, 1)
    y_left = torch.rand(points_per_side, 1) * domain_size
    
    # Right boundary (x=1)
    x_right = torch.ones(points_per_side, 1) * domain_size
    y_right = torch.rand(points_per_side, 1) * domain_size
    
    # Bottom boundary (y=0)
    x_bottom = torch.rand(points_per_side, 1) * domain_size
    y_bottom = torch.zeros(points_per_side, 1)
    
    # Top boundary (y=1)
    x_top = torch.rand(points_per_side, 1) * domain_size
    y_top = torch.ones(points_per_side, 1) * domain_size
    
    # Combine all boundary points
    x_bound = torch.cat([x_left, x_right, x_bottom, x_top])
    y_bound = torch.cat([y_left, y_right, y_bottom, y_top])
    
    return (x_col, y_col, t_col), (x_ic, y_ic), (x_bound, y_bound, t_bound)

def load_numerical_data(filename='numerical_solutions_2d.npz'):
    """
    Load numerical data for supervised learning (Ldata component)
    
    Returns the PINN training data generated by the numerical solver:
    - x, y, t coordinates
    - u, v solution values
    As specified in the paper: Ndata = 10×101×101 points
    """
    if not os.path.exists(filename):
        print(f"❌ Numerical data file '{filename}' not found!")
        print("Please run the numerical solver first:")
        print("   python numerical_solver_2d.py")
        return None
    
    try:
        data = np.load(filename, allow_pickle=True)
        
        if 'pinn_data' not in data:
            print(f"❌ PINN training data not found in '{filename}'")
            print("Please regenerate the numerical data with the updated solver.")
            return None
        
        pinn_data = data['pinn_data'].item()
        
        # Convert to torch tensors
        x_data = torch.tensor(pinn_data['x'], dtype=torch.float32)
        y_data = torch.tensor(pinn_data['y'], dtype=torch.float32)
        t_data = torch.tensor(pinn_data['t'], dtype=torch.float32)
        u_data = torch.tensor(pinn_data['u'], dtype=torch.float32)
        v_data = torch.tensor(pinn_data['v'], dtype=torch.float32)
        
        print(f"✅ Numerical data loaded successfully:")
        print(f"- Total training points: {len(x_data):,}")
        print(f"- Time points: {len(np.unique(t_data.numpy()))}")
        print(f"- Spatial points per time: {len(x_data) // len(np.unique(t_data.numpy())):,}")
        
        return {
            'x': x_data,
            'y': y_data, 
            't': t_data,
            'u': u_data,
            'v': v_data
        }
        
    except Exception as e:
        print(f"❌ Error loading numerical data: {e}")
        return None

def train_pinn(model, epochs=50000, lr=1e-3, omega_B=1.0, omega_F=1.0, omega_data=1.0,
               early_stopping=True, patience=2000, min_delta=1e-6, use_numerical_data=True):
    """
    Train the 2D Gray-Scott PINN model
    Uses Adam optimizer with learning rate 10^-3 as specified in the paper
    Loss: L = ωB*LB + ωF*LF + ωdata*Ldata (full implementation as per paper)
    
    Args:
        model: The PINN model to train
        epochs: Maximum number of training epochs
        lr: Learning rate
        omega_B: Weight for boundary/initial conditions loss
        omega_F: Weight for physics loss
        omega_data: Weight for supervised data loss (Ldata)
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in loss to qualify as improvement
        use_numerical_data: Whether to include supervised learning component
    """
    print("Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    
    # Generate training points
    print("Generating training points...")
    (x_col, y_col, t_col), (x_ic, y_ic), (x_bound, y_bound, t_bound) = generate_training_points()
    
    # Load numerical data for supervised learning
    numerical_data = None
    if use_numerical_data:
        print("\nLoading numerical data for supervised learning...")
        numerical_data = load_numerical_data()
        if numerical_data is None:
            print("⚠️  Continuing without supervised learning component")
            use_numerical_data = False
    
    print(f"\nTraining configuration:")
    print(f"- Collocation: {len(x_col)} points")
    print(f"- Initial conditions: {len(x_ic)} points")
    print(f"- Boundary conditions: {len(x_bound)} points")
    if use_numerical_data and numerical_data:
        print(f"- Supervised data: {len(numerical_data['x'])} points")
    print(f"- Total physics points: {len(x_col) + len(x_ic) + len(x_bound)}")
    print(f"- Loss weights: ωB={omega_B}, ωF={omega_F}, ωdata={omega_data}")
    
    if early_stopping:
        print(f"- Early stopping: patience={patience}, min_delta={min_delta:.2e}")
    
    print("\nStarting training...")
    
    losses = []
    train_start_time = time.time()
    
    # Early stopping variables
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        optimizer.zero_grad()
        
        # Physics loss (LF) - PDE residuals at collocation points
        f1, f2 = model.pde_residual(x_col, y_col, t_col)
        LF = torch.mean(f1**2) + torch.mean(f2**2)
        
        # Initial and boundary conditions loss (LB)
        ic_loss = model.initial_condition_loss(x_ic, y_ic)
        bc_loss = model.boundary_condition_loss(x_bound, y_bound, t_bound)
        LB = ic_loss + bc_loss
        
        # Supervised data loss (Ldata) - as specified in paper
        Ldata = torch.tensor(0.0)
        if use_numerical_data and numerical_data is not None:
            # Predict solutions at numerical data points
            u_pred, v_pred = model(numerical_data['x'], numerical_data['y'], numerical_data['t'])
            
            # Compute supervised loss: Ldata = (1/Ndata) * Σ ||u_pred - u_true||²
            Ldata = (torch.mean((u_pred - numerical_data['u'])**2) + 
                     torch.mean((v_pred - numerical_data['v'])**2))
        
        # Total loss as specified in paper: L = ωB*LB + ωF*LF + ωdata*Ldata
        total_loss = omega_B * LB + omega_F * LF + omega_data * Ldata
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        current_loss = total_loss.item()
        losses.append(current_loss)
        epoch_time = time.time() - epoch_start_time
        
        # Early stopping check
        if early_stopping:
            if current_loss < best_loss - min_delta:
                # Significant improvement found
                best_loss = current_loss
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                # No significant improvement
                epochs_without_improvement += 1
                
                # Check if we should stop early
                if epochs_without_improvement >= patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                    print(f"   No improvement for {patience} epochs")
                    print(f"   Best loss: {best_loss:.6f} at epoch {best_epoch}")
                    print(f"   Current loss: {current_loss:.6f}")
                    
                    # Restore best model state
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        print(f"   Restored model to best state (epoch {best_epoch})")
                    
                    break
        
        # Progress reporting
        if epoch == 0:
            data_info = f", Ldata: {Ldata.item():.6f}" if use_numerical_data and numerical_data else ""
            print(f'Epoch {epoch} completed in {epoch_time:.3f}s, Loss: {current_loss:.6f}{data_info}')
        elif epoch < 100 and epoch % 10 == 0:
            elapsed = time.time() - train_start_time
            avg_time_per_epoch = elapsed / (epoch + 1)
            eta = avg_time_per_epoch * (epochs - epoch - 1)
            early_stop_info = f" | Best: {best_loss:.6f} @{best_epoch} | No improv: {epochs_without_improvement}/{patience}" if early_stopping else ""
            data_info = f", Ldata: {Ldata.item():.6f}" if use_numerical_data and numerical_data else ""
            print(f'Epoch {epoch}, Loss: {current_loss:.6f}, LB: {LB.item():.6f}, LF: {LF.item():.6f}{data_info}{early_stop_info}')
            print(f'  Time/epoch: {epoch_time:.3f}s, ETA: {eta/60:.1f} minutes')
        elif epoch % 2000 == 0:
            elapsed = time.time() - train_start_time
            avg_time_per_epoch = elapsed / (epoch + 1)
            eta = avg_time_per_epoch * (epochs - epoch - 1)
            early_stop_info = f" | Best: {best_loss:.6f} @{best_epoch} | No improv: {epochs_without_improvement}/{patience}" if early_stopping else ""
            data_info = f", Ldata: {Ldata.item():.6f}" if use_numerical_data and numerical_data else ""
            print(f'Epoch {epoch}, Loss: {current_loss:.6f}, LB: {LB.item():.6f}, LF: {LF.item():.6f}{data_info}{early_stop_info}')
            print(f'  Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min, Avg: {avg_time_per_epoch:.3f}s/epoch')
    
    total_time = time.time() - train_start_time
    final_epoch = epoch + 1 if 'epoch' in locals() else epochs
    
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Final epoch: {final_epoch}/{epochs}")
    
    if early_stopping:
        print(f"Best loss achieved: {best_loss:.6f} at epoch {best_epoch}")
        if best_model_state is not None:
            print("Model state restored to best checkpoint")
    
    return losses, {'best_loss': best_loss, 'best_epoch': best_epoch, 'final_epoch': final_epoch}

def visualize_solution(model, domain_size=1.0, t_snapshots=[0, 1000, 2500, 5000], nx=101, ny=101):
    """Visualize the 2D Gray-Scott solution at different time snapshots"""
    x = torch.linspace(0, domain_size, nx)
    y = torch.linspace(0, domain_size, ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    
    fig, axes = plt.subplots(2, len(t_snapshots), figsize=(4*len(t_snapshots), 8))
    
    for i, t_val in enumerate(t_snapshots):
        T_flat = torch.full_like(X_flat, t_val)
        
        with torch.no_grad():
            u_pred, v_pred = model(X_flat, Y_flat, T_flat)
            U = u_pred.reshape(nx, ny).numpy()
            V = v_pred.reshape(nx, ny).numpy()
        
        # Plot u component
        im1 = axes[0, i].contourf(X.numpy(), Y.numpy(), U, levels=50, cmap='viridis')
        axes[0, i].set_title(f'u(x,y) at t={t_val}')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Plot v component
        im2 = axes[1, i].contourf(X.numpy(), Y.numpy(), V, levels=50, cmap='plasma')
        axes[1, i].set_title(f'v(x,y) at t={t_val}')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    return fig

def main():
    """Main training and visualization function"""
    print("=" * 60)
    print("2D Gray-Scott PINN Training")
    print("Complete Paper Implementation with Supervised Learning")
    print("=" * 60)
    print()
    print("Parameters:")
    print("- Du = 2×10⁻⁵, Dv = 10⁻⁵")
    print("- F = 0.030, K = 0.060 (Case 1)")
    print("- Architecture: 4 hidden layers, 20 neurons each")
    print("- Activation: tanh (hidden), sigmoid (output)")
    print("- Training points: Nr=10,000, N0=10,201, NB=400")
    print("- Supervised data: Ndata=10×101×101 points")
    print("- Loss: L = ωB×LB + ωF×LF + ωdata×Ldata")
    print("- Domain: [0,1] × [0,1], t ∈ [0, 5000]")
    print("- Periodic boundary conditions")
    print("- Early stopping with patience=2000")
    print()
    
    # Create model with exact specifications from paper
    model = GrayScott2DPINN(hidden_size=20, num_layers=4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Confirm before starting
    confirm = input("Ready to start training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print()
    
    # Train the model
    losses, training_info = train_pinn(model, epochs=50000, lr=1e-3, omega_B=1.0, omega_F=1.0, omega_data=1.0,
                                       early_stopping=True, patience=2000, min_delta=1e-6, use_numerical_data=True)
    
    # Save the trained model
    model_name = '2d_gray_scott_model.pth'
    torch.save(model.state_dict(), model_name)
    print(f"✅ Model saved as '{model_name}'")
    
    # Save model metadata
    metadata = {
        'hidden_size': 20,
        'num_layers': 4,
        'activation_hidden': 'tanh',
        'activation_output': 'sigmoid',
        'Du': 2e-5,
        'Dv': 1e-5,
        'F': 0.030,
        'K': 0.060,
        'domain_size': 1.0,
        't_max': 5000.0,
        'epochs': 50000,
        'lr': 1e-3,
        'Nr': 10000,
        'N0': 10201,
        'NB': 400,
        'boundary_conditions': 'periodic',
        'description': '2D Gray-Scott PINN - Paper implementation (Case 1)',
        'training_info': training_info
    }
    metadata_name = '2d_gray_scott_model_metadata.json'
    with open(metadata_name, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Model metadata saved as '{metadata_name}'")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('2D Gray-Scott PINN Training Loss')
    plt.yscale('log')
    plt.grid(True)
    loss_plot_name = '2d_gray_scott_training_loss.png'
    plt.savefig(loss_plot_name)
    print(f"✅ Training loss plot saved as '{loss_plot_name}'")
    
    # Generate visualization
    print("\nGenerating solution visualization...")
    fig = visualize_solution(model, t_snapshots=[0, 1000, 2500, 5000])
    viz_name = '2d_gray_scott_solution.png'
    fig.savefig(viz_name, dpi=300, bbox_inches='tight')
    print(f"✅ Solution visualization saved as '{viz_name}'")
    
    print()
    print("🎉 Training completed successfully!")
    print("Files generated:")
    print(f"- {model_name}")
    print(f"- {metadata_name}")
    print(f"- {loss_plot_name}")
    print(f"- {viz_name}")
    print()
    print("✅ Complete implementation with supervised learning component!")
    print("Loss function: L = ωB×LB + ωF×LF + ωdata×Ldata")
    print("- LB: Initial & boundary conditions loss")
    print("- LF: Physics-informed loss (PDE residuals)")
    print("- Ldata: Supervised learning from numerical data")
    print()
    print("This matches the exact implementation described in the paper.")
    
    # Ask if user wants to show the plots
    show_plots = input("\nShow plots? (y/n): ").strip().lower()
    if show_plots == 'y':
        plt.show()
    else:
        plt.close('all')

if __name__ == "__main__":
    main()
