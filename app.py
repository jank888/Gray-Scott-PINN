import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
from gray_scott_pinn import GrayScottPINN

# Set page config
st.set_page_config(
    page_title="Gray-Scott PINN Visualizer",
    page_icon="🌊",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained PINN model"""
    import json
    import os
    
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
            
            st.success(f"✅ Loaded {mode_info} model: {model_file}")
            return model, True
            
        except Exception as e:
            st.warning(f"⚠️ Failed to load {model_file}: {str(e)}")
            continue
    
    st.error("""
    ❌ **No trained model found!** 
    
    Please train a model first by running:
    ```bash
    python gray_scott_pinn.py
    ```
    """)
    return GrayScottPINN(), False

def generate_grid(resolution=64):
    """Generate spatial grid for visualization"""
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    return X.flatten(), Y.flatten()

def predict_pattern(model, x, y, t, F, k):
    """Predict u and v values using the trained PINN"""
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        t_tensor = torch.full_like(x_tensor, t)
        F_tensor = torch.full_like(x_tensor, F)
        k_tensor = torch.full_like(x_tensor, k)
        
        u, v = model(x_tensor, y_tensor, t_tensor, F_tensor, k_tensor)
        return u.numpy().flatten(), v.numpy().flatten()

def create_custom_colormap():
    """Create a custom colormap for visualization"""
    colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']
    return LinearSegmentedColormap.from_list('custom', colors)

def main():
    st.title("🌊 Gray-Scott PINN Visualizer")
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("Parameters")
    
    # Parameter sliders
    F = st.sidebar.slider(
        "Feed Rate (F)", 
        min_value=0.01, 
        max_value=0.11, 
        value=0.055, 
        step=0.001
    )
    
    k = st.sidebar.slider(
        "Kill Rate (k)", 
        min_value=0.01, 
        max_value=0.11, 
        value=0.062, 
        step=0.001
    )
    
    # Time control
    t = st.sidebar.slider(
        "Time", 
        min_value=0.0, 
        max_value=30.0, 
        value=5.0, 
        step=0.2
    )
    
    # Resolution control
    resolution = st.sidebar.selectbox(
        "Grid Resolution", 
        options=[32, 64, 128], 
        index=1
    )
    
    # Animation controls
    st.sidebar.header("Animation")
    animate = st.sidebar.checkbox("Enable Animation")
    
    if animate:
        animation_speed = st.sidebar.slider(
            "Animation Speed", 
            min_value=0.1, 
            max_value=2.0, 
            value=0.5, 
            step=0.1
        )
    
    # Main visualization area
    col1, col2 = st.columns(2)
    
    # Generate grid
    x_grid, y_grid = generate_grid(resolution)
    
    # Create placeholders for plots
    with col1:
        st.subheader("Species U (Activator)")
        plot_placeholder_u = st.empty()
    
    with col2:
        st.subheader("Species V (Inhibitor)")
        plot_placeholder_v = st.empty()
    
    # Create custom colormap
    cmap = create_custom_colormap()
    
    # Animation loop
    if animate:
        for frame in range(100):
            current_time = (frame * animation_speed * 0.3) % 30.0
            
            # Predict patterns
            u_pred, v_pred = predict_pattern(model, x_grid, y_grid, current_time, F, k)
            
            # Reshape for plotting
            u_2d = u_pred.reshape(resolution, resolution)
            v_2d = v_pred.reshape(resolution, resolution)
            
            # Plot U
            fig_u, ax_u = plt.subplots(figsize=(6, 6))
            im_u = ax_u.imshow(u_2d, cmap=cmap, origin='lower', extent=[0, 1, 0, 1])
            ax_u.set_title(f'Species U at t={current_time:.1f}')
            ax_u.set_xlabel('x')
            ax_u.set_ylabel('y')
            
            # Plot V
            fig_v, ax_v = plt.subplots(figsize=(6, 6))
            im_v = ax_v.imshow(v_2d, cmap=cmap, origin='lower', extent=[0, 1, 0, 1])
            ax_v.set_title(f'Species V at t={current_time:.1f}')
            ax_v.set_xlabel('x')
            ax_v.set_ylabel('y')
            
            # Update plots
            with col1:
                plot_placeholder_u.pyplot(fig_u)
            with col2:
                plot_placeholder_v.pyplot(fig_v)
            
            plt.close(fig_u)
            plt.close(fig_v)
            
            time.sleep(0.1)
    else:
        # Static visualization
        u_pred, v_pred = predict_pattern(model, x_grid, y_grid, t, F, k)
        
        # Reshape for plotting
        u_2d = u_pred.reshape(resolution, resolution)
        v_2d = v_pred.reshape(resolution, resolution)
        
        # Plot U
        fig_u, ax_u = plt.subplots(figsize=(6, 6))
        im_u = ax_u.imshow(u_2d, cmap=cmap, origin='lower', extent=[0, 1, 0, 1])
        ax_u.set_title(f'Species U (F={F:.3f}, k={k:.3f}, t={t:.1f})')
        ax_u.set_xlabel('x')
        ax_u.set_ylabel('y')
        
        # Plot V
        fig_v, ax_v = plt.subplots(figsize=(6, 6))
        im_v = ax_v.imshow(v_2d, cmap=cmap, origin='lower', extent=[0, 1, 0, 1])
        ax_v.set_title(f'Species V (F={F:.3f}, k={k:.3f}, t={t:.1f})')
        ax_v.set_xlabel('x')
        ax_v.set_ylabel('y')
        
        # Display plots
        with col1:
            plot_placeholder_u.pyplot(fig_u)
        with col2:
            plot_placeholder_v.pyplot(fig_v)
        
        plt.close(fig_u)
        plt.close(fig_v)

if __name__ == "__main__":
    main() 