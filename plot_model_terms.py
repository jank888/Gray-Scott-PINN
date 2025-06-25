import torch
import numpy as np
import matplotlib.pyplot as plt
from gray_scott_pinn import GrayScottPINN
import json
import os

def load_model():
    """Load the trained PINN model"""
    # Try loading models in order of preference
    model_files = ['gray_scott_pinn_model.pth', 'gray_scott_pinn_model_fast.pth']
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            continue
            
        try:
            # Try to load metadata first
            metadata_file = model_file.replace('.pth', '_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                config = {
                    'hidden_size': metadata['hidden_size'],
                    'num_layers': metadata['num_layers']
                }
                mode_info = metadata['mode']
            else:
                # Fallback to filename-based detection
                if 'fast' in model_file:
                    config = {'hidden_size': 50, 'num_layers': 4}
                    mode_info = 'fast'
                else:
                    config = {'hidden_size': 100, 'num_layers': 5}
                    mode_info = 'standard'
            
            # Create model with correct architecture
            model = GrayScottPINN(**config)
            model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
            model.eval()
            
            print(f"✅ Loaded {mode_info} model: {model_file}")
            print(f"📊 Architecture: {config['hidden_size']} neurons, {config['num_layers']} layers")
            return model, True
            
        except Exception as e:
            print(f"⚠️ Failed to load {model_file}: {str(e)}")
            continue
    
    print("❌ No trained model found!")
    return None, False

def compute_model_terms(model, x, y, t, F, k):
    """Compute all terms in the Gray-Scott equations"""
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    F = F.clone().detach()
    k = k.clone().detach()
    
    # Get model predictions with gradient tracking
    with torch.enable_grad():
        u, v = model(x, y, t, F, k)
    
    # Compute spatial derivatives
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
    
    # Compute temporal derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    
    # Compute key terms
    laplacian_u = u_xx + u_yy
    laplacian_v = v_xx + v_yy
    reaction_term = u * v**2
    
    # Gray-Scott equation terms
    # du/dt = Du * ∇²u - uv² + F(1-u)
    diffusion_u = model.Du * laplacian_u
    feed_u = F * (1 - u)
    
    # dv/dt = Dv * ∇²v + uv² - (F+k)v
    diffusion_v = model.Dv * laplacian_v
    kill_v = -(F + k) * v
    
    return {
        'u': u.detach(),
        'v': v.detach(),
        'u_t': u_t.detach(),
        'v_t': v_t.detach(),
        'laplacian_u': laplacian_u.detach(),
        'laplacian_v': laplacian_v.detach(),
        'reaction_term': reaction_term.detach(),
        'diffusion_u': diffusion_u.detach(),
        'diffusion_v': diffusion_v.detach(),
        'feed_u': feed_u.detach(),
        'kill_v': kill_v.detach()
    }

def generate_test_grid(resolution=64, t_val=5.0, F_val=0.055, k_val=0.062):
    """Generate a test grid for visualization"""
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_flat = torch.full_like(x_flat, t_val)
    F_flat = torch.full_like(x_flat, F_val)
    k_flat = torch.full_like(x_flat, k_val)
    
    return x_flat, y_flat, t_flat, F_flat, k_flat, (resolution, resolution)

def plot_terms(terms, shape, title_prefix="", save_prefix="model_terms"):
    """Plot all the computed terms"""
    resolution = shape[0]
    
    # Define terms to plot with their descriptions
    plot_terms = [
        ('u', 'Species U concentration'),
        ('v', 'Species V concentration'),
        ('u_t', '∂u/∂t (time derivative of u)'),
        ('v_t', '∂v/∂t (time derivative of v)'),
        ('laplacian_u', '∇²u (Laplacian of u)'),
        ('laplacian_v', '∇²v (Laplacian of v)'),
        ('reaction_term', 'uv² (reaction term)'),
        ('diffusion_u', 'Du·∇²u (diffusion term for u)'),
        ('diffusion_v', 'Dv·∇²v (diffusion term for v)'),
        ('feed_u', 'F(1-u) (feed term for u)'),
        ('kill_v', '-(F+k)v (kill term for v)')
    ]
    
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (term_name, description) in enumerate(plot_terms):
        if i >= len(axes):
            break
            
        ax = axes[i]
        term_data = terms[term_name].numpy().reshape(resolution, resolution)
        
        # Plot with appropriate colormap
        if 'u' in term_name or 'v' in term_name:
            # Concentration fields - use viridis
            im = ax.imshow(term_data, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
        else:
            # Derivatives and other terms - use RdBu_r (centered on 0)
            vmax = np.abs(term_data).max()
            vmin = -vmax
            im = ax.imshow(term_data, origin='lower', extent=[0, 1, 0, 1], 
                          cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        ax.set_title(f'{description}', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Print statistics
        mean_val = np.mean(term_data)
        std_val = np.std(term_data)
        min_val = np.min(term_data)
        max_val = np.max(term_data)
        
        print(f"{term_name:15s}: mean={mean_val:8.4f}, std={std_val:8.4f}, "
              f"min={min_val:8.4f}, max={max_val:8.4f}")
        
        # Check for degeneracy
        if std_val < 1e-6:
            print(f"  ⚠️  WARNING: {term_name} appears to be spatially constant (std={std_val:.2e})")
        if abs(mean_val) < 1e-8 and abs(max_val) < 1e-8:
            print(f"  ⚠️  WARNING: {term_name} appears to be identically zero")
    
    # Hide unused subplots
    for i in range(len(plot_terms), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{title_prefix}Gray-Scott PINN Model Terms Analysis', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    filename = f"{save_prefix}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved as '{filename}'")
    
    return fig

def analyze_multiple_times(model, times=[0.0, 2.5, 5.0, 10.0, 15.0]):
    """Analyze model terms at multiple time points"""
    print("\n" + "="*60)
    print("🔍 ANALYZING MODEL TERMS AT MULTIPLE TIME POINTS")
    print("="*60)
    
    for t_val in times:
        print(f"\n📅 TIME t = {t_val}")
        print("-" * 40)
        
        # Generate test grid
        x, y, t, F, k, shape = generate_test_grid(resolution=32, t_val=t_val)
        
        # Compute terms
        terms = compute_model_terms(model, x, y, t, F, k)
        
        # Quick statistics check
        for term_name in ['u', 'v', 'u_t', 'v_t', 'laplacian_u', 'laplacian_v', 'reaction_term']:
            term_data = terms[term_name].numpy()
            std_val = np.std(term_data)
            mean_val = np.mean(term_data)
            
            if std_val < 1e-6:
                print(f"  ⚠️  {term_name}: CONSTANT (std={std_val:.2e})")
            elif abs(mean_val) < 1e-8 and np.abs(term_data).max() < 1e-8:
                print(f"  ⚠️  {term_name}: ZERO (max={np.abs(term_data).max():.2e})")
            else:
                print(f"  ✅ {term_name}: OK (std={std_val:.4f}, range=[{np.min(term_data):.4f}, {np.max(term_data):.4f}])")

def main():
    print("🔍 Gray-Scott PINN Model Terms Analysis")
    print("="*50)
    
    # Load model
    model, success = load_model()
    if not success:
        return
    
    print("\n🧮 Computing model terms...")
    
    # Test at a few different conditions
    test_cases = [
        (5.0, 0.055, 0.062, "Standard Parameters"),
        (1.0, 0.035, 0.065, "Early Time + Different Parameters"),
        (10.0, 0.075, 0.045, "Late Time + Chaos Parameters")
    ]
    
    for i, (t_val, F_val, k_val, description) in enumerate(test_cases):
        print(f"\n📊 Case {i+1}: {description}")
        print(f"    t={t_val}, F={F_val}, k={k_val}")
        print("-" * 50)
        
        # Generate test grid
        x, y, t, F, k, shape = generate_test_grid(
            resolution=64, t_val=t_val, F_val=F_val, k_val=k_val
        )
        
        # Compute terms
        terms = compute_model_terms(model, x, y, t, F, k)
        
        # Plot terms
        title_prefix = f"{description} (t={t_val}) - "
        save_prefix = f"model_terms_case_{i+1}"
        fig = plot_terms(terms, shape, title_prefix, save_prefix)
        
        # Show plot for first case only
        if i == 0:
            plt.show()
        else:
            plt.close(fig)
    
    # Analyze temporal evolution
    analyze_multiple_times(model)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey things to check:")
    print("1. Are u and v showing spatial patterns (not flat)?")
    print("2. Are time derivatives (u_t, v_t) non-zero and spatially varying?")
    print("3. Are Laplacians (∇²u, ∇²v) showing curvature information?")
    print("4. Is the reaction term (uv²) non-zero where u and v overlap?")
    print("5. Are diffusion terms proportional to Laplacians?")
    print("\nIf any terms are flat/constant, the model may have collapsed!")

if __name__ == "__main__":
    main()