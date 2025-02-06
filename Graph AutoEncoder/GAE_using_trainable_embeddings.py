import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import NNConv
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

######################
# DATA LOADING
######################
train_data = pd.read_csv('train_data.csv', index_col=0)
val_data   = pd.read_csv('val_data.csv', index_col=0)
test_data  = pd.read_csv('test_data.csv', index_col=0)

# Build edge indices and edge attributes for training
edge_index = torch.tensor(train_data[['srcip_index', 'dstip_index']].values.T, dtype=torch.long)
edge_attrs = torch.tensor(train_data.drop(columns=['srcip', 'dstip', 'srcip_index', 'dstip_index']).values, dtype=torch.float)

# Build global node indices (all nodes that appear in training)
global_node_indices = set(train_data['srcip_index']).union(set(train_data['dstip_index']))
global_node_indices = torch.tensor(sorted(global_node_indices), dtype=torch.long)
global_num_nodes = len(global_node_indices)

# Build test edges and attributes
test_edge_index = torch.tensor(test_data[['srcip_index', 'dstip_index']].values.T, dtype=torch.long)
test_edge_attrs = torch.tensor(test_data.drop(columns=['srcip', 'dstip', 'srcip_index', 'dstip_index', 'Label']).values, dtype=torch.float)

# Create PyG Data objects for training, validation, and testing
train_graph_data = Data(x=None, edge_index=edge_index, edge_attr=edge_attrs, y=None)
test_graph_data  = Data(x=None, edge_index=test_edge_index, edge_attr=test_edge_attrs, y=test_data['Label'])

train_graph_data.num_nodes = global_num_nodes
test_graph_data.num_nodes  = global_num_nodes

# Create validation Data object
val_edge_index = torch.tensor(val_data[['srcip_index', 'dstip_index']].values.T, dtype=torch.long)
val_edge_attrs = torch.tensor(val_data.drop(columns=['srcip', 'dstip', 'srcip_index', 'dstip_index']).values, dtype=torch.float)
val_graph_data = Data(x=None, edge_index=val_edge_index, edge_attr=val_edge_attrs, y=None)
val_graph_data.num_nodes = global_num_nodes

######################
# GRAPH AUTOENCODER (USING STRICTLY TRAINABLE EMBEDDINGS)
######################
# Define a simple node embedding layer.
class NodeEmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        # The embeddings will be randomly initialized and are trainable by default.
        
    def forward(self):
        return self.node_embeddings.weight  # returns a tensor of shape (num_nodes, embedding_dim)

class GraphAutoencoder(nn.Module):
    def __init__(self, num_nodes, embedding_dim, latent_dim, edge_dim):
        """
        num_nodes: total number of nodes in the graph.
        embedding_dim: dimension of the trainable node embeddings.
        latent_dim: dimension of the latent (compressed) representation.
        edge_dim: number of edge features.
        """
        super().__init__()
        # Trainable node embeddings.
        self.node_embeddings = NodeEmbeddingLayer(num_nodes, embedding_dim)
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        
        # Encoder: using NNConv to aggregate neighbor information.
        self.edge_weight_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim * latent_dim)  # Output weights for NNConv
        )
        self.encoder_conv = NNConv(
            in_channels=self.embedding_dim,
            out_channels=latent_dim,
            nn=self.edge_weight_encoder,
            aggr='mean'
        )
        
        # Decoder: reconstruct edge features from concatenated latent node embeddings.
        self.edge_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, edge_dim)
        )
    
    def encode(self, edge_index, edge_attr):
        # Use the trainable node embeddings.
        x = self.node_embeddings()  # shape: (num_nodes, embedding_dim)
        return self.encoder_conv(x, edge_index, edge_attr)
    
    def decode(self, latent_x, edge_index):
        src, tgt = edge_index
        edge_input = torch.cat([latent_x[src], latent_x[tgt]], dim=1)
        return self.edge_predictor(edge_input)
    
    def forward(self, edge_index, edge_attr):
        latent_x = self.encode(edge_index, edge_attr)
        reconstructed_edge_attr = self.decode(latent_x, edge_index)
        return reconstructed_edge_attr

# Hyperparameters for the autoencoder.
embedding_dim = 16  # length of (embedding) vector representing each node
latent_dim = 8      
edge_dim = train_graph_data.num_edge_features  # number of edge features

# Instantiate the model (with strictly trainable embeddings).
model = GraphAutoencoder(global_num_nodes, embedding_dim, latent_dim, edge_dim).to(device)

optimizer_model = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()

# Move data to device.
edge_index = train_graph_data.edge_index.to(device)
edge_attrs = train_graph_data.edge_attr.to(device)
val_edge_index = val_graph_data.edge_index.to(device)
val_edge_attrs = val_graph_data.edge_attr.to(device)
test_edge_index = test_graph_data.edge_index.to(device)
test_edge_attrs = test_graph_data.edge_attr.to(device)

######################
# TRAINING LOOP (and recording training and validation loss))
######################
model.train()
training_losses = []  # record training loss per epoch
val_losses = []       # record validation loss per epoch

num_epochs = 50
for epoch in range(num_epochs):
    optimizer_model.zero_grad()
    # Forward pass on training data.
    reconstructed_edge_attr = model(edge_index, edge_attrs)
    loss = loss_function(reconstructed_edge_attr, edge_attrs)
    loss.backward()
    optimizer_model.step()
    
    training_losses.append(loss.item())
    
    # Compute validation loss.
    model.eval()
    with torch.no_grad():
        val_reconstructed = model(val_edge_index, val_edge_attrs)
        val_loss = loss_function(val_reconstructed, val_edge_attrs)
    val_losses.append(val_loss.item())
    model.train()  # Switch back to training mode
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Plot and save the training vs. validation loss curve.
plt.figure(figsize=(12, 8))
plt.plot(range(num_epochs), training_losses, label="Training Loss", linewidth=2, color='blue')
plt.plot(range(num_epochs), val_losses, label="Validation Loss", linewidth=2, linestyle="dashed", color='red')
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.title("Training and Validation Loss Curve", fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("learnable_embeddings_loss_curve.pdf", dpi=300)
plt.show()

######################
# THRESHOLD SELECTION (based on training errors)
######################
model.eval()
with torch.no_grad():
    train_reconstructed = model(edge_index, edge_attrs)
train_errors = torch.abs(edge_attrs - train_reconstructed).cpu().numpy()
train_errors_aggregated = train_errors.mean(axis=1)
threshold = np.percentile(train_errors_aggregated, 95)
print("Selected Threshold (95th percentile of training errors):", threshold)

######################
# TESTING PHASE
######################
with torch.no_grad():
    test_reconstructed = model(test_edge_index, test_edge_attrs)
    test_loss = loss_function(test_reconstructed, test_edge_attrs)
print(f"Test Loss: {test_loss.item():.4f}")

edge_errors = torch.abs(test_edge_attrs - test_reconstructed).cpu().numpy()
edge_errors_aggregated = edge_errors.mean(axis=1)

# Generate binary predictions using the threshold.
preds = (edge_errors_aggregated > threshold).astype(int)
true_labels = test_data['Label'].values

cm = confusion_matrix(true_labels, preds)
cr = classification_report(true_labels, preds)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)

######################
# PLOTTING PRECISION-RECALL CURVE
######################
precision, recall, _ = precision_recall_curve(true_labels, edge_errors_aggregated)
pr_auc = auc(recall, precision)
plt.figure(figsize=(12, 8))
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})', color='blue', linewidth=2)
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.title('Precision-Recall Curve', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("learnable_embeddings_precision_recall_curve.pdf", dpi=300)
plt.show()

######################
# PLOTTING ROC CURVE
######################
fpr, tpr, _ = roc_curve(true_labels, edge_errors_aggregated)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curve', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("learnable_embeddings_roc_curve.pdf", dpi=300)
plt.show()

######################
# NETWORKX SUBGRAPH VISUALIZATION
######################
# Sample a subset (5%) of edges from the test set.
sample_indices = np.random.choice(len(test_edge_attrs), size=int(0.05 * len(test_edge_attrs)), replace=False)
sample_edges = test_edge_index[:, sample_indices].cpu().T
sample_scores = edge_errors_aggregated[sample_indices]
sample_labels = preds[sample_indices]

# Create a directed graph using NetworkX.
G = nx.DiGraph()
for i, (src, dst) in enumerate(sample_edges):
    G.add_edge(src.item(), dst.item(), anomaly=sample_labels[i], error=sample_scores[i])
    
# Compute a layout for visualization.
pos = nx.spring_layout(G, seed=42)

# Plot WITHOUT node labels
plt.figure(figsize=(12, 8))
normal_edges = [(u, v) for u, v, d in G.edges(data=True) if d['anomaly'] == 0]
nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='grey', alpha=0.3)
anomalous_edges = [(u, v) for u, v, d in G.edges(data=True) if d['anomaly'] == 1]
nx.draw_networkx_edges(G, pos, edgelist=anomalous_edges, edge_color='red', width=2, alpha=0.7)
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='blue')
plt.title('Anomalous Edges (Red) vs. Normal Edges (Grey)', fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("learnable_embeddings_anomalous_vs_normal_Edges_no_labels.pdf", dpi=300, format='pdf')
plt.show()

# Plot WITH node labels
plt.figure(figsize=(12, 8))
nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='grey', alpha=0.3, width=1)
nx.draw_networkx_edges(G, pos, edgelist=anomalous_edges, edge_color='red', width=2, alpha=0.7)
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='blue', alpha=0.8)
nx.draw_networkx_labels(G, pos, font_size=12, font_color='white')
plt.title('Anomalous Edges (Red) vs. Normal Edges (Grey)', fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("learnable_embeddings_anomalous_vs_normal_Edges_with_labels.pdf", dpi=300, format='pdf')
plt.show()
