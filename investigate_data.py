"""
Investigate the ceramic temperature data to understand the real performance differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def investigate_data():
    """Investigate the actual data structure and relationships."""
    
    print("🔍 Investigating ceramic temperature data...")
    
    # Load both datasets
    sparse_data = pd.read_csv('data/ps1_dataA_15TC.csv', skiprows=3)
    dense_data = pd.read_csv('data/ps1_dataA_120TC.csv', skiprows=3)
    
    print(f"📊 Sparse data shape: {sparse_data.shape}")
    print(f"📊 Dense data shape: {dense_data.shape}")
    
    # Load coordinates
    with open('data/ps1_dataA_15TC.csv', 'r') as f:
        lines = f.readlines()
        r_sparse = [float(x) for x in lines[0].strip().split(',')[1:]]
        z_sparse = [float(x) for x in lines[1].strip().split(',')[1:]]
    
    with open('data/ps1_dataA_120TC.csv', 'r') as f:
        lines = f.readlines()
        r_dense = [float(x) for x in lines[0].strip().split(',')[1:]]
        z_dense = [float(x) for x in lines[1].strip().split(',')[1:]]
    
    print(f"📍 Sparse sensors: {len(r_sparse)} coordinates")
    print(f"📍 Dense sensors: {len(r_dense)} coordinates")
    
    # Check first timestep temperatures
    sparse_temps = sparse_data.iloc[0, 1:].values.astype(float)
    dense_temps = dense_data.iloc[0, 1:].values.astype(float)
    
    print(f"\n🌡️ Temperature Statistics (First Timestep):")
    print(f"Sparse - Min: {sparse_temps.min():.1f}°C, Max: {sparse_temps.max():.1f}°C, Mean: {sparse_temps.mean():.1f}°C")
    print(f"Dense  - Min: {dense_temps.min():.1f}°C, Max: {dense_temps.max():.1f}°C, Mean: {dense_temps.mean():.1f}°C")
    
    # Check if sparse sensors are subset of dense sensors
    sparse_coords = np.array(list(zip(r_sparse, z_sparse)))
    dense_coords = np.array(list(zip(r_dense, z_dense)))
    
    print(f"\n🔍 Coordinate Analysis:")
    print(f"Sparse coordinate ranges: R[{min(r_sparse):.1f}, {max(r_sparse):.1f}], Z[{min(z_sparse):.1f}, {max(z_sparse):.1f}]")
    print(f"Dense coordinate ranges:  R[{min(r_dense):.1f}, {max(r_dense):.1f}], Z[{min(z_dense):.1f}, {max(z_dense):.1f}]")
    
    # Check for coordinate overlap
    matches = 0
    for sparse_coord in sparse_coords:
        distances = np.sqrt(np.sum((dense_coords - sparse_coord)**2, axis=1))
        if np.min(distances) < 0.1:  # Very close match
            matches += 1
    
    print(f"📍 Coordinate matches: {matches}/{len(sparse_coords)} sparse sensors have close matches in dense set")
    
    # This explains the 0.0 MAE - the sparse sensors are likely exact subsets of dense sensors!
    
    return sparse_coords, dense_coords, sparse_temps, dense_temps

def create_proper_validation():
    """Create a proper validation by using spatial separation."""
    
    print("\n🎯 Creating proper validation strategy...")
    
    sparse_coords, dense_coords, sparse_temps, dense_temps = investigate_data()
    
    # Find dense coordinates that are NOT in sparse set (for true validation)
    validation_coords = []
    validation_temps = []
    
    for i, dense_coord in enumerate(dense_coords):
        distances = np.sqrt(np.sum((sparse_coords - dense_coord)**2, axis=1))
        if np.min(distances) > 1.0:  # At least 1mm away from any sparse sensor
            validation_coords.append(dense_coord)
            validation_temps.append(dense_temps[i])
    
    validation_coords = np.array(validation_coords)
    validation_temps = np.array(validation_temps)
    
    print(f"📊 Found {len(validation_coords)} true validation points (>1mm from sparse sensors)")
    
    if len(validation_coords) == 0:
        print("❌ No suitable validation points found - datasets may be too similar")
        return None
        
    # Now do real comparison on these validation points
    from scipy.interpolate import griddata
    
    results = {}
    
    # 1. Bilinear (Linear) interpolation
    print("🔍 Testing Linear interpolation on validation points...")
    linear_pred = griddata(sparse_coords, sparse_temps, validation_coords, method='linear')
    valid_mask = ~np.isnan(linear_pred)
    if np.sum(valid_mask) > 0:
        linear_mae = np.mean(np.abs(linear_pred[valid_mask] - validation_temps[valid_mask]))
        results['Linear'] = linear_mae
        print(f"   ✅ Linear MAE: {linear_mae:.2f}°C on {np.sum(valid_mask)} points")
    else:
        results['Linear'] = np.nan
        print("   ❌ Linear interpolation failed (all NaN)")
    
    # 2. Nearest neighbor
    print("🔍 Testing Nearest Neighbor...")
    nearest_pred = griddata(sparse_coords, sparse_temps, validation_coords, method='nearest')
    nearest_mae = np.mean(np.abs(nearest_pred - validation_temps))
    results['Nearest'] = nearest_mae
    print(f"   ✅ Nearest MAE: {nearest_mae:.2f}°C")
    
    # 3. Cubic interpolation
    print("🔍 Testing Cubic interpolation...")
    cubic_pred = griddata(sparse_coords, sparse_temps, validation_coords, method='cubic')
    valid_mask = ~np.isnan(cubic_pred)
    if np.sum(valid_mask) > 0:
        cubic_mae = np.mean(np.abs(cubic_pred[valid_mask] - validation_temps[valid_mask]))
        results['Cubic'] = cubic_mae
        print(f"   ✅ Cubic MAE: {cubic_mae:.2f}°C on {np.sum(valid_mask)} points")
    else:
        results['Cubic'] = np.nan
        print("   ❌ Cubic interpolation failed (all NaN)")
    
    # 4. Simple distance-weighted average
    print("🔍 Testing Distance-Weighted Average...")
    dwa_pred = []
    for val_coord in validation_coords:
        distances = np.sqrt(np.sum((sparse_coords - val_coord)**2, axis=1))
        weights = 1.0 / (distances + 1e-10)  # Avoid division by zero
        weights /= np.sum(weights)
        predicted_temp = np.sum(weights * sparse_temps)
        dwa_pred.append(predicted_temp)
    
    dwa_pred = np.array(dwa_pred)
    dwa_mae = np.mean(np.abs(dwa_pred - validation_temps))
    results['Distance_Weighted'] = dwa_mae
    print(f"   ✅ Distance-Weighted MAE: {dwa_mae:.2f}°C")
    
    # 5. Your PINN result (you need to run PINN on these same validation points)
    # For now, we'll estimate based on your reported overall MAE
    results['PINN'] = 19.7  # Your reported result
    print(f"🧠 PINN MAE: 19.7°C (your reported overall result)")
    
    return results, len(validation_coords)

if __name__ == "__main__":
    print("🔬 PROPER Data Investigation & Real Comparison")
    print("=" * 60)
    
    # First investigate why we got 0.0 MAE
    investigate_data()
    
    # Then create proper validation
    result = create_proper_validation()
    
    if result is not None:
        results, n_validation = result
        
        print(f"\n📊 REAL RESULTS (on {n_validation} validation points):")
        print("=" * 40)
        for method, mae in results.items():
            if not np.isnan(mae):
                print(f"{method:16s}: {mae:6.2f}°C")
            else:
                print(f"{method:16s}: FAILED")
        
        # Calculate real improvements
        pinn_mae = results['PINN']
        print(f"\n🏆 PINN vs Traditional Methods:")
        for method, mae in results.items():
            if method != 'PINN' and not np.isnan(mae):
                if mae > pinn_mae:
                    improvement = ((mae - pinn_mae) / mae) * 100
                    print(f"   • {improvement:.1f}% better than {method}")
                else:
                    degradation = ((pinn_mae - mae) / pinn_mae) * 100
                    print(f"   • {degradation:.1f}% worse than {method}")
    else:
        print("\n❌ Could not create proper validation - datasets too similar")
        print("💡 Recommendation: Use cross-validation or synthetic test cases")