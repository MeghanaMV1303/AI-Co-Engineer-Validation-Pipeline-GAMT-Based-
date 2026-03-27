import os
import torch
from torch_geometric.data import Data
import numpy as np
import hashlib

def generate_synthetic_graph(filename="default.step"):
    """
    Generates a PyTorch Geometric Data object with varying geometry 
    based on the filename hash to simulate different CAD parts.
    """
    # Deterministic generation based on filename
    seed = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 10000
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    filename_lower = filename.lower()
    
    # 1. Varying Node Count (faces)
    if "cube" in filename_lower:
        num_nodes = 6
    elif "cylinder" in filename_lower:
        num_nodes = 4
    elif "assembly" in filename_lower:
        num_nodes = 24
    else:
        # Between 8 and 60 faces
        num_nodes = torch.randint(8, 60, (1,)).item()
        
    # 2. Node features: [area, z_normal, is_vertical]
    x_features = []
    for i in range(num_nodes):
        area = float(torch.randint(5, 500, (1,)).item())
        z_norm = torch.rand(1).item() * 2 - 1.0 # -1 to 1
        is_vert = 1.0 if abs(z_norm) < 0.15 else 0.0
        x_features.append([area, abs(z_norm), is_vert])
        
    x = torch.tensor(x_features, dtype=torch.float)
    
    # 3. Edges: Connect nodes to form a realistic face graph
    edge_index_list = []
    edge_attr_list = []
    
    for i in range(num_nodes - 1):
        edge_index_list.extend([[i, i+1], [i+1, i]])
        angle = float(torch.randint(60, 120, (1,)).item())
        convexity = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        edge_attr_list.extend([[10.0, angle, convexity], [10.0, angle, convexity]])
        
    # Add random extra edges for density
    extra_edges = num_nodes // 2
    for _ in range(extra_edges):
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        if u != v:
            edge_index_list.extend([[u, v], [v, u]])
            angle = float(torch.randint(30, 150, (1,)).item())
            edge_attr_list.extend([[15.0, angle, -1.0], [15.0, angle, -1.0]])
            
    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    # 4. Multi-task Labels [draft_violation, thin_wall, sharp_edge, risk_score]
    y = torch.zeros((num_nodes, 4), dtype=torch.float)
    
    # Draft violation if is_vertical == 1
    y[:, 0] = x[:, 2] 
    
    # Thin wall if area < 20
    y[:, 1] = (x[:, 0] < 20).float()
    
    # Sharp edge randomly (20% chance)
    y[:, 2] = (torch.rand(num_nodes) > 0.8).float()
    
    # Risk score
    y[:, 3] = (y[:, 0]*0.5 + y[:, 1]*0.3 + y[:, 2]*0.2).clamp(0, 1)
        
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Optional explicitly stored filename for debug
    data.filename = filename
    return data

def create_dataset(num_samples=200, save_dir="data"):
    """Creates a synthetic dataset with diverse graphs."""
    os.makedirs(save_dir, exist_ok=True)
    dataset = []
    for i in range(num_samples):
        # Generate random "filenames" to create variety
        fname = f"part_{i}_{torch.randint(0, 10000, (1,)).item()}.step"
        data = generate_synthetic_graph(filename=fname)
        dataset.append(data)
        
    torch.save(dataset, os.path.join(save_dir, "cubes_dataset.pt"))
    print(f"Generated {num_samples} synthetic graphs. Saved to {save_dir}/cubes_dataset.pt")
    
if __name__ == "__main__":
    create_dataset()
