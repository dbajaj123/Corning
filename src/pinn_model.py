"""
Physics-Informed Neural Network for Ceramic Temperature Interpolation
Corning Future Innovation Program 2025

This module implements a Physics-Informed Neural Network (PINN) for predicting
temperature distribution in ceramic tiles during manufacturing with sparse sensor data.

Author: [Your Name]
Created: 2025
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any


class TemperaturePINN(nn.Module):
    """
    Physics-Informed Neural Network for temperature interpolation in ceramic manufacturing.
    
    This network enforces the heat equation as a physics constraint while learning
    from sparse sensor data to predict temperature at any spatial location.
    
    The network architecture consists of fully connected layers with ReLU activations
    and enforces the steady-state heat equation: ∇²T + GE/k = 0
    
    Attributes:
        layers (nn.ModuleList): Neural network layers
        physics_weight (float): Weight for physics loss term
        data_weight (float): Weight for data fitting term
    """
    
    def __init__(self, 
                 input_dim: int = 2, 
                 hidden_layers: list = [50, 100, 100, 50], 
                 output_dim: int = 1,
                 physics_weight: float = 1.0,
                 data_weight: float = 1.0):
        """
        Initialize the PINN model.
        
        Args:
            input_dim (int): Input dimension (x, y coordinates)
            hidden_layers (list): List of hidden layer sizes
            output_dim (int): Output dimension (temperature)
            physics_weight (float): Weight for physics loss component
            data_weight (float): Weight for data fitting component
        """
        super(TemperaturePINN, self).__init__()
        
        self.physics_weight = physics_weight
        self.data_weight = data_weight
        
        # Build network architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input coordinates (batch_size, 2)
            
        Returns:
            torch.Tensor: Predicted temperatures (batch_size, 1)
        """
        return self.layers(x)
    
    def compute_physics_loss(self, 
                           x_physics: torch.Tensor, 
                           thermal_conductivity: float = 1.0,
                           generation_rate: float = 0.0) -> torch.Tensor:
        """
        Compute physics-informed loss based on heat equation residual.
        
        Enforces the steady-state heat equation: ∇²T + GE/k = 0
        
        Args:
            x_physics (torch.Tensor): Collocation points for physics evaluation
            thermal_conductivity (float): Material thermal conductivity
            generation_rate (float): Heat generation rate
            
        Returns:
            torch.Tensor: Physics loss (mean squared residual)
        """
        # Enable gradient computation
        x_physics = x_physics.clone().detach().requires_grad_(True)
        
        # Forward pass
        T = self.forward(x_physics)
        
        # Compute first derivatives
        T_x = torch.autograd.grad(T.sum(), x_physics, create_graph=True)[0][:, 0:1]
        T_y = torch.autograd.grad(T.sum(), x_physics, create_graph=True)[0][:, 1:2]
        
        # Compute second derivatives (Laplacian)
        T_xx = torch.autograd.grad(T_x.sum(), x_physics, create_graph=True)[0][:, 0:1]
        T_yy = torch.autograd.grad(T_y.sum(), x_physics, create_graph=True)[0][:, 1:2]
        
        # Heat equation residual: ∇²T + GE/k = 0
        laplacian = T_xx + T_yy
        heat_equation_residual = laplacian + generation_rate / thermal_conductivity
        
        # Return mean squared residual
        return torch.mean(heat_equation_residual ** 2)
    
    def compute_data_loss(self, 
                         x_data: torch.Tensor, 
                         y_data: torch.Tensor) -> torch.Tensor:
        """
        Compute data fitting loss for known sensor measurements.
        
        Args:
            x_data (torch.Tensor): Sensor coordinates
            y_data (torch.Tensor): Sensor temperature measurements
            
        Returns:
            torch.Tensor: Data fitting loss (MSE)
        """
        predictions = self.forward(x_data)
        return nn.MSELoss()(predictions, y_data)
    
    def compute_total_loss(self, 
                          x_data: torch.Tensor, 
                          y_data: torch.Tensor,
                          x_physics: torch.Tensor,
                          thermal_conductivity: float = 1.0,
                          generation_rate: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Compute total PINN loss combining data and physics terms.
        
        Args:
            x_data (torch.Tensor): Sensor coordinates
            y_data (torch.Tensor): Sensor measurements
            x_physics (torch.Tensor): Collocation points for physics
            thermal_conductivity (float): Material thermal conductivity
            generation_rate (float): Heat generation rate
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing individual and total losses
        """
        # Compute individual loss components
        data_loss = self.compute_data_loss(x_data, y_data)
        physics_loss = self.compute_physics_loss(
            x_physics, thermal_conductivity, generation_rate
        )
        
        # Combine losses
        total_loss = (self.data_weight * data_loss + 
                     self.physics_weight * physics_loss)
        
        return {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss
        }
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions for given coordinates.
        
        Args:
            x (torch.Tensor): Input coordinates
            
        Returns:
            np.ndarray: Predicted temperatures
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()
    
    def save_model(self, filepath: str):
        """Save model state dict to file."""
        torch.save(self.state_dict(), filepath)
        
    def load_model(self, filepath: str):
        """Load model state dict from file."""
        self.load_state_dict(torch.load(filepath, map_location='cpu'))


class PINNTrainer:
    """
    Training class for Physics-Informed Neural Network.
    
    Handles the complete training pipeline including data preparation,
    loss computation, optimization, and validation.
    """
    
    def __init__(self, 
                 model: TemperaturePINN,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        """
        Initialize the PINN trainer.
        
        Args:
            model (TemperaturePINN): PINN model to train
            learning_rate (float): Optimizer learning rate
            device (str): Device for computation ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_history = {
            'total': [], 'data': [], 'physics': []
        }
        
    def prepare_training_data(self, 
                            sensor_coords: np.ndarray,
                            sensor_temps: np.ndarray,
                            domain_bounds: Tuple[float, float, float, float],
                            n_physics_points: int = 1000) -> Tuple[torch.Tensor, ...]:
        """
        Prepare training data including sensor measurements and physics collocation points.
        
        Args:
            sensor_coords (np.ndarray): Sensor coordinates (N, 2)
            sensor_temps (np.ndarray): Sensor temperatures (N, 1)
            domain_bounds (tuple): Domain boundaries (x_min, x_max, y_min, y_max)
            n_physics_points (int): Number of collocation points for physics
            
        Returns:
            Tuple of tensors: (x_data, y_data, x_physics)
        """
        # Convert sensor data to tensors
        x_data = torch.FloatTensor(sensor_coords).to(self.device)
        y_data = torch.FloatTensor(sensor_temps.reshape(-1, 1)).to(self.device)
        
        # Generate random collocation points for physics
        x_min, x_max, y_min, y_max = domain_bounds
        x_physics = torch.zeros((n_physics_points, 2)).to(self.device)
        x_physics[:, 0] = x_min + (x_max - x_min) * torch.rand(n_physics_points)
        x_physics[:, 1] = y_min + (y_max - y_min) * torch.rand(n_physics_points)
        
        return x_data, y_data, x_physics
    
    def train_epoch(self, 
                   x_data: torch.Tensor,
                   y_data: torch.Tensor, 
                   x_physics: torch.Tensor,
                   thermal_conductivity: float = 1.0,
                   generation_rate: float = 0.0) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            x_data (torch.Tensor): Sensor coordinates
            y_data (torch.Tensor): Sensor measurements
            x_physics (torch.Tensor): Physics collocation points
            thermal_conductivity (float): Material thermal conductivity
            generation_rate (float): Heat generation rate
            
        Returns:
            Dict[str, float]: Loss values for the epoch
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute losses
        losses = self.model.compute_total_loss(
            x_data, y_data, x_physics, thermal_conductivity, generation_rate
        )
        
        # Backward pass
        losses['total_loss'].backward()
        self.optimizer.step()
        
        # Convert to float for logging
        epoch_losses = {k: v.item() for k, v in losses.items()}
        
        # Update loss history
        self.loss_history['total'].append(epoch_losses['total_loss'])
        self.loss_history['data'].append(epoch_losses['data_loss'])
        self.loss_history['physics'].append(epoch_losses['physics_loss'])
        
        return epoch_losses
    
    def train(self, 
             sensor_coords: np.ndarray,
             sensor_temps: np.ndarray,
             domain_bounds: Tuple[float, float, float, float],
             epochs: int = 1000,
             n_physics_points: int = 1000,
             thermal_conductivity: float = 1.0,
             generation_rate: float = 0.0,
             print_interval: int = 100) -> Dict[str, list]:
        """
        Complete training loop.
        
        Args:
            sensor_coords (np.ndarray): Sensor coordinates
            sensor_temps (np.ndarray): Sensor temperatures
            domain_bounds (tuple): Domain boundaries
            epochs (int): Number of training epochs
            n_physics_points (int): Number of physics collocation points
            thermal_conductivity (float): Material thermal conductivity
            generation_rate (float): Heat generation rate
            print_interval (int): Interval for printing progress
            
        Returns:
            Dict[str, list]: Training loss history
        """
        # Prepare training data
        x_data, y_data, x_physics = self.prepare_training_data(
            sensor_coords, sensor_temps, domain_bounds, n_physics_points
        )
        
        print(f"Training PINN for {epochs} epochs...")
        print(f"Data points: {len(x_data)}, Physics points: {len(x_physics)}")
        
        for epoch in range(epochs):
            # Train one epoch
            losses = self.train_epoch(
                x_data, y_data, x_physics, thermal_conductivity, generation_rate
            )
            
            # Print progress
            if (epoch + 1) % print_interval == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Total: {losses['total_loss']:.6f}, "
                      f"Data: {losses['data_loss']:.6f}, "
                      f"Physics: {losses['physics_loss']:.6f}")
        
        print("Training completed!")
        return self.loss_history
    
    def plot_loss_history(self):
        """Plot training loss history."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(self.loss_history['total'], label='Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].grid(True)
        axes[0].set_yscale('log')
        
        axes[1].plot(self.loss_history['data'], label='Data Loss', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Data Loss')
        axes[1].grid(True)
        axes[1].set_yscale('log')
        
        axes[2].plot(self.loss_history['physics'], label='Physics Loss', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Physics Loss')
        axes[2].grid(True)
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        plt.show()


def load_ceramic_data(sparse_file: str, dense_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load ceramic temperature data from CSV files.
    
    Args:
        sparse_file (str): Path to sparse sensor data (15TC)
        dense_file (str, optional): Path to dense sensor data (120TC)
        
    Returns:
        Dict containing loaded data and coordinates
    """
    # Load sparse data
    sparse_data = pd.read_csv(sparse_file, skiprows=3)
    
    # Extract coordinates from header
    with open(sparse_file, 'r') as f:
        lines = f.readlines()
        r_coords = [float(x) for x in lines[0].strip().split(',')[1:]]
        z_coords = [float(x) for x in lines[1].strip().split(',')[1:]]
    
    sparse_coords = np.array(list(zip(r_coords, z_coords)))
    
    result = {
        'sparse_data': sparse_data,
        'sparse_coords': sparse_coords,
        'r_coords': np.array(r_coords),
        'z_coords': np.array(z_coords)
    }
    
    # Load dense data if provided
    if dense_file:
        dense_data = pd.read_csv(dense_file, skiprows=3)
        
        with open(dense_file, 'r') as f:
            lines = f.readlines()
            r_coords_dense = [float(x) for x in lines[0].strip().split(',')[1:]]
            z_coords_dense = [float(x) for x in lines[1].strip().split(',')[1:]]
        
        dense_coords = np.array(list(zip(r_coords_dense, z_coords_dense)))
        
        result.update({
            'dense_data': dense_data,
            'dense_coords': dense_coords,
            'r_coords_dense': np.array(r_coords_dense),
            'z_coords_dense': np.array(z_coords_dense)
        })
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Physics-Informed Neural Network for Ceramic Temperature Interpolation")
    print("Corning Future Innovation Program 2025")
    print("\nThis module provides a complete PINN implementation for temperature prediction.")
    print("Use in Jupyter notebook or import as a module for custom applications.")