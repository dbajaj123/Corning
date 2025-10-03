"""
Simple Visualization of Temperature Field Problem
================================================

Creates basic plots to visualize the sensor layouts and temperature distributions
without complex dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_temperature_data(file_path):
    """Load temperature data from CSV files."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        r_coords = [float(x) for x in lines[0].strip().split(',')[1:] if x.strip()]
        z_coords = [float(x) for x in lines[1].strip().split(',')[1:] if x.strip()]
    
    coords = np.array(list(zip(r_coords, z_coords)))
    return coords

def create_sensor_layout_comparison():
    """Create visualization comparing sparse vs dense sensor layouts."""
    
    # Load coordinate data
    project_dir = Path.cwd()
    sparse_coords = load_temperature_data(project_dir / 'ps1_dataA_15TC.csv')
    dense_coords = load_temperature_data(project_dir / 'ps1_dataA_120TC.csv')
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Dense sensor layout (120 sensors - ground truth)
    ax1.scatter(dense_coords[:, 0], dense_coords[:, 1], 
               c='blue', s=50, alpha=0.7, marker='o', label='120 Sensors')
    ax1.set_title('Dense Sensor Layout\n(Ground Truth - 120 Sensors)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('R coordinate (m)', fontsize=12)
    ax1.set_ylabel('Z coordinate (m)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_aspect('equal')
    
    # Plot 2: Sparse sensor layout (15 sensors)
    ax2.scatter(sparse_coords[:, 0], sparse_coords[:, 1], 
               c='red', s=120, alpha=0.8, marker='s', label='15 Sensors')
    ax2.set_title('Sparse Sensor Layout\n(87.5% Reduction - 15 Sensors)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('R coordinate (m)', fontsize=12)
    ax2.set_ylabel('Z coordinate (m)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_aspect('equal')
    
    # Plot 3: Overlay comparison
    ax3.scatter(dense_coords[:, 0], dense_coords[:, 1], 
               c='lightblue', s=30, alpha=0.5, marker='o', label='Dense (120)')
    ax3.scatter(sparse_coords[:, 0], sparse_coords[:, 1], 
               c='red', s=120, alpha=0.9, marker='s', 
               edgecolors='darkred', linewidth=1, label='Sparse (15)')
    ax3.set_title('Sensor Layout Comparison\n(Sparse vs Dense)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('R coordinate (m)', fontsize=12)
    ax3.set_ylabel('Z coordinate (m)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    ax3.set_aspect('equal')
    
    # Add problem context
    fig.suptitle('Temperature Field Reconstruction Problem\n' + 
                 'Ceramic Manufacturing Process Monitoring', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add technical details
    textstr = ('Domain: 2D Cylindrical Coordinates\\n'
               'Temperature Range: 27°C - 1,185°C\\n'
               'Time Duration: 25 hours\\n'
               'Sensor Reduction: 87.5%')
    
    fig.text(0.02, 0.02, textstr, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # Save figure removed as requested
    
    plt.show()
    
    return fig

def create_performance_comparison():
    """Create bar chart comparing different methods."""
    
    # Performance data from analysis
    methods = ['Linear\\nInterpolation', 'Cubic\\nInterpolation', 'Nearest\\nNeighbor', 'PINN\\n(Physics-Informed)']
    mae_values = [75.2, 65.7, 86.2, 19.7]
    colors = ['lightcoral', 'lightsalmon', 'lightpink', 'lightgreen']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mean Absolute Error comparison
    bars = ax1.bar(methods, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Temperature Prediction Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error (°C)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, mae_values):
        height = bar.get_height()
        ax1.annotate(f'{value:.1f}°C',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Highlight PINN performance
    bars[3].set_color('darkgreen')
    bars[3].set_alpha(1.0)
    
    # Plot 2: Accuracy improvement factors
    improvement_factors = [mae_val/19.7 for mae_val in mae_values]
    bars2 = ax2.bar(methods, improvement_factors, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Accuracy Improvement vs PINN', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Error Factor (relative to PINN)', fontsize=12)
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='PINN Baseline')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Add value labels
    for bar, value in zip(bars2, improvement_factors):
        height = bar.get_height()
        if value == 1.0:
            label = 'Baseline'
        else:
            label = f'{value:.1f}x worse'
        ax2.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Highlight PINN
    bars2[3].set_color('darkgreen')
    bars2[3].set_alpha(1.0)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Performance comparison saved as 'performance_comparison.png'")
    
    plt.show()
    
    return fig

def main():
    """Main visualization function."""
    
    print("="*60)
    print("TEMPERATURE FIELD RECONSTRUCTION VISUALIZATION")
    print("="*60)
    print("Creating visualizations for:")
    print("1. Sensor layout comparison (sparse vs dense)")
    print("2. Performance comparison (bilinear vs PINN)")
    print("="*60)
    
    try:
        # Create sensor layout visualization
        print("\\n📍 Creating sensor layout comparison...")
        create_sensor_layout_comparison()
        
        # Create performance comparison
        print("\\n📈 Creating performance comparison...")
        create_performance_comparison()
        
        print("\\n✅ All visualizations completed successfully!")
        print("\\nFiles created:")

        print("- performance_comparison.png")
        
        print("\\n🎯 KEY INSIGHTS FROM VISUALIZATION:")
        print("- Sparse sensors (15) strategically placed across domain")
        print("- PINN achieves 3.3x better accuracy than best interpolation")
        print("- Physics constraints critical for sparse sensor scenarios")
        print("- 87.5% sensor reduction with superior performance")
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        print("Please ensure matplotlib is available and data files exist.")

if __name__ == "__main__":
    main()