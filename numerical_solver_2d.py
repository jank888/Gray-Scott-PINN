import numpy as np
import matplotlib.pyplot as plt
import json
import time

class GrayScott2DNumericalSolver:
    def __init__(self, nx=101, ny=101, Lx=1.0, Ly=1.0, 
                 Du=2e-5, Dv=1e-5, F=0.030, K=0.060):
        """
        2D Gray-Scott Numerical Solver using Finite Difference Method
        
        Parameters as specified in the paper:
        - Second order accurate in space
        - First order accurate in time
        - Grid: 101 × 101 points
        - Domain: [0,1] × [0,1]
        - Periodic boundary conditions
        """
        
        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        
        # Spatial discretization
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # Create spatial grids
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Gray-Scott parameters from paper
        self.Du = Du  # Diffusion coefficient for u
        self.Dv = Dv  # Diffusion coefficient for v  
        self.F = F    # Feed rate
        self.K = K    # Kill rate
        
        # Initialize solution arrays
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        
        # Time stepping parameters
        self.dt = None
        self.current_time = 0.0
        
        print(f"Initialized 2D Gray-Scott solver:")
        print(f"- Grid: {nx} × {ny}")
        print(f"- Domain: [{0}, {Lx}] × [{0}, {Ly}]")
        print(f"- dx = {self.dx:.6f}, dy = {self.dy:.6f}")
        print(f"- Parameters: Du={Du:.2e}, Dv={Dv:.2e}, F={F:.3f}, K={K:.3f}")
    
    def set_initial_conditions(self):
        """
        Set initial conditions as specified in the paper:
        - Trivial state (u = 1, v = 0) everywhere
        - Small square perturbation at center (u = 1/2, v = 1/4)
        """
        # Initialize to trivial state
        self.u.fill(1.0)  # u = 1 everywhere
        self.v.fill(0.0)  # v = 0 everywhere
        
        # Define central square perturbation
        center_x, center_y = self.Lx/2, self.Ly/2
        square_size = 0.1  # Small square size
        
        # Find indices for the central square
        mask_x = np.abs(self.X - center_x) <= square_size/2
        mask_y = np.abs(self.Y - center_y) <= square_size/2
        mask = mask_x & mask_y
        
        # Apply perturbation
        self.u[mask] = 0.5   # u = 1/2 in central square
        self.v[mask] = 0.25  # v = 1/4 in central square
        
        self.current_time = 0.0
        
        print("Initial conditions set:")
        print(f"- Trivial state: u=1, v=0")
        print(f"- Central perturbation: u=0.5, v=0.25 in {square_size}×{square_size} square")
    
    def calculate_stable_dt(self, safety_factor=0.1):
        """
        Calculate stable time step based on diffusion stability criterion
        For explicit schemes: dt ≤ min(dx²,dy²)/(4*max(Du,Dv))
        """
        dt_diffusion = min(self.dx**2, self.dy**2) / (4 * max(self.Du, self.Dv))
        dt_stable = safety_factor * dt_diffusion
        
        print(f"Stability analysis:")
        print(f"- Diffusion limit: dt ≤ {dt_diffusion:.6f}")
        print(f"- Chosen dt: {dt_stable:.6f} (safety factor: {safety_factor})")
        
        return dt_stable
    
    def apply_periodic_bc(self, field):
        """
        Apply periodic boundary conditions to a 2D field
        This ensures u(0,y,t) = u(1,y,t) and u(x,0,t) = u(x,1,t)
        """
        field_bc = field.copy()
        
        # For periodic BCs, opposite boundaries must be equal
        # Left boundary (x=0) equals right boundary (x=1)
        # Bottom boundary (y=0) equals top boundary (y=1)
        
        return field_bc  # Return original field for now, periodic BC handled in Laplacian
    
    def laplacian_2d_periodic(self, field):
        """
        Compute 2D Laplacian using second-order central differences with periodic BCs
        ∇²u = ∂²u/∂x² + ∂²u/∂y²
        
        Second order accurate in space as specified in the paper
        """
        laplacian = np.zeros_like(field)
        
        # For periodic boundary conditions, we use modular arithmetic
        for i in range(self.nx):
            for j in range(self.ny):
                # Previous and next indices with periodic wrapping
                i_prev = (i - 1) % self.nx
                i_next = (i + 1) % self.nx
                j_prev = (j - 1) % self.ny
                j_next = (j + 1) % self.ny
                
                # Second-order central differences
                d2u_dx2 = (field[i_next, j] - 2*field[i, j] + field[i_prev, j]) / self.dx**2
                d2u_dy2 = (field[i, j_next] - 2*field[i, j] + field[i, j_prev]) / self.dy**2
                
                laplacian[i, j] = d2u_dx2 + d2u_dy2
        
        return laplacian
    
    def rhs_gray_scott(self, u, v):
        """
        Compute right-hand side of Gray-Scott equations:
        ∂u/∂t = Du*∇²u - uv² + F(1-u)
        ∂v/∂t = Dv*∇²v + uv² - (F+K)v
        """
        # Compute Laplacians with periodic boundary conditions
        lap_u = self.laplacian_2d_periodic(u)
        lap_v = self.laplacian_2d_periodic(v)
        
        # Reaction terms
        reaction = u * v**2
        
        # Gray-Scott equations
        dudt = self.Du * lap_u - reaction + self.F * (1 - u)
        dvdt = self.Dv * lap_v + reaction - (self.F + self.K) * v
        
        return dudt, dvdt
    
    def step_forward_euler(self, dt):
        """
        Take one time step using Forward Euler (first order in time)
        As specified in the paper: "first order accurate in time"
        """
        # Compute RHS at current time
        dudt, dvdt = self.rhs_gray_scott(self.u, self.v)
        
        # Forward Euler update
        self.u = self.u + dt * dudt
        self.v = self.v + dt * dvdt
        
        # Update time
        self.current_time += dt
    
    def solve(self, t_final=5000.0, save_times=None, dt=None, verbose=True):
        """
        Solve the Gray-Scott system from t=0 to t=t_final
        
        Parameters:
        - t_final: Final simulation time
        - save_times: List of times to save solutions (default: paper specification)
        - dt: Time step (if None, automatically calculated)
        - verbose: Print progress
        """
        if save_times is None:
            # Default save times from paper: t ∈ {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000}
            save_times = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        
        if dt is None:
            dt = self.calculate_stable_dt()
        
        self.dt = dt
        
        # Storage for solutions
        solutions = {}
        times_saved = []
        
        # Initial condition (save t=0 if requested)
        if 0.0 in save_times:
            solutions[0.0] = {'u': self.u.copy(), 'v': self.v.copy()}
            times_saved.append(0.0)
        
        print(f"\nStarting numerical integration:")
        print(f"- Final time: {t_final}")
        print(f"- Time step: {dt:.6f}")
        print(f"- Total steps: {int(t_final/dt):,}")
        print(f"- Save times: {save_times}")
        
        start_time = time.time()
        step_count = 0
        
        # Main time loop
        while self.current_time < t_final:
            # Take time step
            self.step_forward_euler(dt)
            step_count += 1
            
            # Save solution at specified times
            for save_t in save_times:
                if abs(self.current_time - save_t) < dt/2 and save_t not in times_saved:
                    solutions[save_t] = {'u': self.u.copy(), 'v': self.v.copy()}
                    times_saved.append(save_t)
                    if verbose:
                        elapsed = time.time() - start_time
                        progress = self.current_time / t_final * 100
                        print(f"  t = {save_t:6.0f} ({progress:5.1f}%) - "
                              f"Step {step_count:,} - Elapsed: {elapsed/60:.1f}min")
            
            # Progress updates for long runs
            if verbose and step_count % 100000 == 0:
                elapsed = time.time() - start_time
                progress = self.current_time / t_final * 100
                eta = elapsed / progress * (100 - progress) if progress > 0 else 0
                print(f"  t = {self.current_time:8.1f} ({progress:5.1f}%) - "
                      f"Step {step_count:,} - ETA: {eta/60:.1f}min")
        
        total_time = time.time() - start_time
        print(f"\nIntegration completed in {total_time/60:.1f} minutes")
        print(f"- Total steps: {step_count:,}")
        print(f"- Solutions saved at {len(times_saved)} time points")
        
        return solutions, times_saved
    
    def generate_pinn_training_data(self, solutions):
        """
        Generate training data for PINN supervised learning (Ldata component)
        
        Returns flattened arrays of coordinates and solution values
        suitable for PINN training as specified in the paper:
        Ndata = 10×101×101 points (10 time steps, 101×101 spatial grid)
        """
        print("\nGenerating PINN training data...")
        
        # Prepare arrays for PINN training
        x_data = []
        y_data = []
        t_data = []
        u_data = []
        v_data = []
        
        for t in sorted(solutions.keys()):
            sol = solutions[t]
            
            # Flatten spatial grids
            x_flat = self.X.flatten()
            y_flat = self.Y.flatten()
            t_flat = np.full_like(x_flat, t)
            u_flat = sol['u'].flatten()
            v_flat = sol['v'].flatten()
            
            # Append to data arrays
            x_data.extend(x_flat)
            y_data.extend(y_flat)
            t_data.extend(t_flat)
            u_data.extend(u_flat)
            v_data.extend(v_flat)
        
        # Convert to numpy arrays
        x_data = np.array(x_data).reshape(-1, 1)
        y_data = np.array(y_data).reshape(-1, 1)
        t_data = np.array(t_data).reshape(-1, 1)
        u_data = np.array(u_data).reshape(-1, 1)
        v_data = np.array(v_data).reshape(-1, 1)
        
        print(f"Generated PINN training data:")
        print(f"- Total points: {len(x_data):,}")
        print(f"- Time points: {len(solutions)}")
        print(f"- Points per time: {len(x_data) // len(solutions):,}")
        
        return {
            'x': x_data,
            'y': y_data,
            't': t_data,
            'u': u_data,
            'v': v_data
        }
    
    def save_solutions(self, solutions, filename='numerical_solutions_2d.npz'):
        """Save numerical solutions to file"""
        # Prepare data for saving
        save_data = {
            'times': list(solutions.keys()),
            'x': self.x,
            'y': self.y,
            'X': self.X,
            'Y': self.Y,
            'parameters': {
                'Du': self.Du,
                'Dv': self.Dv,
                'F': self.F,
                'K': self.K,
                'nx': self.nx,
                'ny': self.ny,
                'Lx': self.Lx,
                'Ly': self.Ly,
                'dx': self.dx,
                'dy': self.dy,
                'dt': self.dt
            }
        }
        
        # Add solution arrays
        for t, sol in solutions.items():
            save_data[f'u_{t}'] = sol['u']
            save_data[f'v_{t}'] = sol['v']
        
        # Add PINN training data
        pinn_data = self.generate_pinn_training_data(solutions)
        save_data['pinn_data'] = pinn_data
        
        np.savez_compressed(filename, **save_data)
        print(f"✅ Solutions saved to '{filename}'")
        print(f"✅ PINN training data included")
    
    def load_solutions(self, filename='numerical_solutions_2d.npz'):
        """Load numerical solutions from file"""
        data = np.load(filename, allow_pickle=True)
        
        # Load parameters
        params = data['parameters'].item()
        times = data['times']
        
        # Reconstruct solutions dictionary
        solutions = {}
        for t in times:
            solutions[float(t)] = {
                'u': data[f'u_{t}'],
                'v': data[f'v_{t}']
            }
        
        # Load PINN training data if available
        pinn_data = None
        if 'pinn_data' in data:
            pinn_data = data['pinn_data'].item()
        
        print(f"✅ Solutions loaded from '{filename}'")
        print(f"- {len(solutions)} time points")
        print(f"- Grid: {params['nx']} × {params['ny']}")
        if pinn_data:
            print(f"- PINN training data: {len(pinn_data['x']):,} points")
        
        return solutions, times, params, pinn_data

def visualize_solutions(solutions, times, x, y, X, Y, save_plots=True):
    """Visualize the numerical solutions"""
    n_times = min(len(times), 5)  # Show at most 5 time snapshots
    selected_times = times[:n_times] if len(times) <= 5 else [times[0], times[2], times[4], times[6], times[-1]]
    
    fig, axes = plt.subplots(2, n_times, figsize=(3*n_times, 6))
    
    if n_times == 1:
        axes = axes.reshape(2, 1)
    
    for i, t in enumerate(selected_times):
        if t in solutions:
            u = solutions[t]['u']
            v = solutions[t]['v']
            
            # Plot u
            im1 = axes[0, i].contourf(X, Y, u, levels=50, cmap='viridis')
            axes[0, i].set_title(f'u(x,y) at t={t}')
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Plot v
            im2 = axes[1, i].contourf(X, Y, v, levels=50, cmap='plasma')
            axes[1, i].set_title(f'v(x,y) at t={t}')
            axes[1, i].set_xlabel('x')
            axes[1, i].set_ylabel('y')
            plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('numerical_solutions_2d.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'numerical_solutions_2d.png'")
    
    return fig

def main():
    """Main function to run the numerical solver"""
    print("=" * 60)
    print("2D Gray-Scott Numerical Solver")
    print("Finite Difference Method (FDM)")
    print("Paper Implementation for PINN Supervised Learning")
    print("=" * 60)
    print()
    
    # Create solver with paper specifications
    solver = GrayScott2DNumericalSolver(
        nx=101, ny=101,          # 101 × 101 grid
        Lx=1.0, Ly=1.0,          # Unit domain
        Du=2e-5, Dv=1e-5,        # Diffusion coefficients
        F=0.030, K=0.060         # Case 1 parameters
    )
    
    # Set initial conditions
    solver.set_initial_conditions()
    
    # Confirm before starting
    confirm = input("Ready to start numerical integration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Integration cancelled.")
        return
    
    print()
    
    # Solve the system
    # Save times from paper: t ∈ {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000}
    save_times = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    solutions, times_saved = solver.solve(
        t_final=5000.0,
        save_times=save_times,
        verbose=True
    )
    
    # Save solutions with PINN training data
    solver.save_solutions(solutions, 'numerical_solutions_2d.npz')
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = visualize_solutions(solutions, times_saved, solver.x, solver.y, solver.X, solver.Y)
    
    print()
    print("🎉 Numerical integration completed successfully!")
    print("Files generated:")
    print("- numerical_solutions_2d.npz (solution data + PINN training data)")
    print("- numerical_solutions_2d.png (visualization)")
    print()
    print("This data can now be used for the supervised component (Ldata) in the PINN model.")
    print("The numerical solutions are generated using:")
    print("- Second-order accurate finite differences in space")
    print("- First-order accurate Forward Euler in time")
    print("- Periodic boundary conditions")
    print(f"- Same parameters as PINN: Du={solver.Du:.2e}, Dv={solver.Dv:.2e}, F={solver.F:.3f}, K={solver.K:.3f}")
    
    # Ask if user wants to show the plot
    show_plot = input("\nShow visualization? (y/n): ").strip().lower()
    if show_plot == 'y':
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main() 