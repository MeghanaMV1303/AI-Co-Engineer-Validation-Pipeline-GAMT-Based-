import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.loader import DataLoader

class GAMT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        # PyG TransformerConv takes node features, edge indices, and edge attributes
        self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = TransformerConv(hidden_channels, out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # First Graph Convolution layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second Graph Convolution layer (Output layer)
        x = self.conv2(x, edge_index, edge_attr)
        
        # Multi-task output layer: we use Sigmoid for multi-label and risk score prediction
        return torch.sigmoid(x)

def train_model(dataset_path="data/cubes_dataset.pt", epochs=50):
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please run synthetic_data.py first.")
        return

    # Load data
    dataset = torch.load(dataset_path, weights_only=False)
    
    # Split 80/20 train/test
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Initialize model
    # in_channels=3 [area, z_normal, is_vertical]
    # out_channels=4 [draft_violation, thin_wall, sharp_edge, risk_score]
    # edge_dim=3 [length, angle, convexity]
    model = GAMT(in_channels=3, hidden_channels=16, out_channels=4, edge_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Loss function for multi-task (MSE works well for [0,1] scores/labels combined)
    criterion = torch.nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            out = model.forward(batch.x, batch.edge_index, batch.edge_attr)
            
            # Compute loss per node
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model.forward(batch.x, batch.edge_index, batch.edge_attr)
            total_loss += criterion(out, batch.y).item()
            
    print(f"Final Validation Loss: {total_loss:.4f}")
    
    # Save the trained weights
    os.makedirs("models", exist_ok=True)
    model_path = "models/draft_gnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained GAMT model to {model_path}")

if __name__ == "__main__":
    train_model()
