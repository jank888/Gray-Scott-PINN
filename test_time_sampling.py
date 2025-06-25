#!/usr/bin/env python3
"""
Test script to visualize the dense time sampling distribution
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_dense_time_samples(n_samples, t_max=15.0, decay_rate=2.5):
    """
    Generate time samples with exponentially decreasing density from t=0
    More natural distribution for physics learning
    """
    # Generate uniform random samples
    uniform_samples = torch.rand(n_samples, 1)
    
    # Transform using exponential distribution
    # This creates natural exponential decay from t=0
    t_samples = -torch.log(1 - uniform_samples * (1 - torch.exp(torch.tensor(-decay_rate)))) / decay_rate * t_max
    
    # Clamp to ensure we stay within [0, t_max]
    t_samples = torch.clamp(t_samples, 0, t_max)
    
    return t_samples

def test_time_sampling():
    """Compare uniform vs dense time sampling"""
    n_samples = 5000
    
    # Generate both distributions
    t_uniform = torch.rand(n_samples, 1) * 15.0  # Old uniform method
    t_dense = generate_dense_time_samples(n_samples)  # New dense method
    
    # Create histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Uniform distribution
    axes[0].hist(t_uniform.numpy(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0].set_title('Uniform Time Sampling (Old)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim(0, 15)
    axes[0].grid(True, alpha=0.3)
    
    # Dense distribution
    axes[1].hist(t_dense.numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1].set_title('Dense Time Sampling (New)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(0, 15)
    axes[1].grid(True, alpha=0.3)
    
    # Overlay comparison
    axes[2].hist(t_uniform.numpy(), bins=50, alpha=0.5, color='red', label='Uniform', edgecolor='black')
    axes[2].hist(t_dense.numpy(), bins=50, alpha=0.5, color='blue', label='Dense', edgecolor='black')
    axes[2].set_title('Comparison')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xlim(0, 15)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("📊 TIME SAMPLING COMPARISON")
    print("=" * 50)
    print(f"Total samples: {n_samples}")
    print()
    
    # Early time analysis (t < 3.75 = t_max/4)
    early_threshold = 3.75
    uniform_early = (t_uniform < early_threshold).sum().item()
    dense_early = (t_dense < early_threshold).sum().item()
    
    print(f"Samples in early time (t < {early_threshold}):")
    print(f"  Uniform: {uniform_early} ({100*uniform_early/n_samples:.1f}%)")
    print(f"  Dense:   {dense_early} ({100*dense_early/n_samples:.1f}%)")
    print(f"  Improvement: {dense_early/uniform_early:.1f}x more samples in early time")
    print()
    
    # Very early time analysis (t < 1.0)
    very_early_threshold = 1.0
    uniform_very_early = (t_uniform < very_early_threshold).sum().item()
    dense_very_early = (t_dense < very_early_threshold).sum().item()
    
    print(f"Samples in very early time (t < {very_early_threshold}):")
    print(f"  Uniform: {uniform_very_early} ({100*uniform_very_early/n_samples:.1f}%)")
    print(f"  Dense:   {dense_very_early} ({100*dense_very_early/n_samples:.1f}%)")
    print(f"  Improvement: {dense_very_early/uniform_very_early:.1f}x more samples in very early time")
    print()
    
    print("🎯 BENEFITS:")
    print("✅ Better learning of initial condition dynamics")
    print("✅ More accurate pattern formation onset")
    print("✅ Improved stability in early time evolution")
    print("✅ Still covers full time range [0, 15]")

def test_decay_rates():
    """Test different decay rates for exponential sampling"""
    n_samples = 5000
    decay_rates = [0.5, 1.0, 2.0, 4.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, decay_rate in enumerate(decay_rates):
        t_samples = generate_dense_time_samples(n_samples, decay_rate=decay_rate)
        
        axes[i].hist(t_samples.numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[i].set_title(f'Decay Rate = {decay_rate}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlim(0, 15)
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        early_count = (t_samples < 3.75).sum().item()
        very_early_count = (t_samples < 1.0).sum().item()
        axes[i].text(0.6, 0.9, f'< 3.75: {100*early_count/n_samples:.1f}%\n< 1.0: {100*very_early_count/n_samples:.1f}%', 
                    transform=axes[i].transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 DECAY RATE COMPARISON")
    print("=" * 50)
    print("Decay Rate | < 1.0s  | < 3.75s | Description")
    print("-" * 50)
    for decay_rate in decay_rates:
        t_samples = generate_dense_time_samples(n_samples, decay_rate=decay_rate)
        early_pct = 100 * (t_samples < 3.75).sum().item() / n_samples
        very_early_pct = 100 * (t_samples < 1.0).sum().item() / n_samples
        
        if decay_rate == 0.5:
            desc = "Very gradual"
        elif decay_rate == 1.0:
            desc = "Moderate"
        elif decay_rate == 2.0:
            desc = "Strong (recommended)"
        else:
            desc = "Very strong"
            
        print(f"{decay_rate:^10} | {very_early_pct:5.1f}% | {early_pct:6.1f}% | {desc}")

if __name__ == "__main__":
    test_time_sampling()
    print("\n" + "="*60)
    test_decay_rates() 