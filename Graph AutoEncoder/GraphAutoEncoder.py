import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.nn import NNConv
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_data = pd.read_csv('train_data.csv', index_col=0)
test_data = pd.read_csv('test_data.csv', index_col=0)

edge_index = torch.tensor(train_data[['srcip_index', 'dstip_index']].values.T)
edge_attrs = torch.tensor(train_data.drop(columns=['srcip', 'dstip', 'srcip_index', 'dstip_index']).values, dtype=torch.float)

global_node_indices = set(train_data['srcip_index']).union(set(train_data['dstip_index']))
global_node_indices = torch.tensor(sorted(global_node_indices), dtype=torch.long)
global_num_nodes = len(global_node_indices)

test_edge_index = torch.tensor(test_data[['srcip_index', 'dstip_index']].values.T)
test_edge_attrs = torch.tensor(test_data.drop(columns=['srcip', 'dstip', 'srcip_index', 'dstip_index', 'Label']).values, dtype=torch.float)

train_graph_data = Data(x=None, edge_index=edge_index , edge_attr=edge_attrs, y=None)
test_graph_data = Data(x=None, edge_index=test_edge_index , edge_attr=test_edge_attrs, y=test_data['Label'])

train_graph_data.num_nodes = global_num_nodes
test_graph_data.num_nodes = global_num_nodes


# from torch_geometric.utils import to_networkx
# import networkx as nx
# import matplotlib.pyplot as plt

# # Convert the PyTorch Geometric train_graph_data into a standard NetworkX directed graph
# train_nx_graph = to_networkx(
#     train_graph_data,
#     to_undirected=False,      # Keep it directed
#     remove_self_loops=True    # or False if you want to preserve self-loops
# )

# # Create an empty MultiDiGraph
# train_multi_graph = nx.MultiDiGraph()

# # Add nodes and their attributes from the converted graph
# train_multi_graph.add_nodes_from(train_nx_graph.nodes(data=True))

# # Add edges and their attributes
# train_multi_graph.add_edges_from(train_nx_graph.edges(data=True))

# # Now visualize the multi-graph
# plt.figure(figsize=(10, 10))
# pos = nx.spring_layout(train_multi_graph, seed=42)  # Layout for clarity/reproducibility

# nx.draw_networkx_nodes(train_multi_graph, pos, node_size=300)
# nx.draw_networkx_labels(train_multi_graph, pos, font_size=10)
# nx.draw_networkx_edges(train_multi_graph, pos, arrowstyle='->', arrowsize=10)

# plt.title("Training Graph as a NetworkX MultiDiGraph")
# plt.axis("off")
# plt.show()

# # Convert the PyTorch Geometric test_graph_data to a standard NetworkX directed graph
# test_nx_graph = to_networkx(
#     test_graph_data,
#     to_undirected=False,
#     remove_self_loops=True
# )

# test_multi_graph = nx.MultiDiGraph()
# test_multi_graph.add_nodes_from(test_nx_graph.nodes(data=True))
# test_multi_graph.add_edges_from(test_nx_graph.edges(data=True))

# # Visualization
# plt.figure(figsize=(10, 10))
# pos = nx.spring_layout(test_multi_graph, seed=42)

# nx.draw_networkx_nodes(test_multi_graph, pos, node_size=300)
# nx.draw_networkx_labels(test_multi_graph, pos, font_size=10)
# nx.draw_networkx_edges(test_multi_graph, pos, arrowstyle='->', arrowsize=10)

# plt.title("Test Graph as a NetworkX MultiDiGraph")
# plt.axis("off")
# plt.show()


# Define Node2Vec parameters
node2vec = Node2Vec(
    edge_index=edge_index,    # Graph structure (edges)
    embedding_dim=16,         # Size of node embeddings
    walk_length=50,           # Length of each random walk
    context_size=5,           # Sliding window size
    walks_per_node=10,        # Number of random walks per node
    p=2,                      # Bias for DFS-like behavior
    q=0.5,                    # Bias for exploring distant nodes
    num_negative_samples=5,   # Number of fake pairs for contrast
    sparse=True               # Efficient gradient updates
)

# Move Node2Vec to GPU (if available)
node2vec = node2vec.to(device)

batch_size = 16  # Number of nodes to process per batch during training

# Train Node2Vec
optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=0.01)
for epoch in range(500):
    optimizer.zero_grad()
    pos_rw, neg_rw = node2vec.sample(batch_size)  # Generate positive and negative random walks
    
    # Move the walks to the same device as the model
    pos_rw = pos_rw.to(device)
    neg_rw = neg_rw.to(device)

    loss = node2vec.loss(pos_rw, neg_rw)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Node2Vec Training Epoch {epoch}, Loss: {loss.item()}")


pretrained_node_embeddings = node2vec.embedding.weight.data

class NodeEmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, embedding_dim, pretrained_embeddings=None):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        
        if pretrained_embeddings is not None:   # Load pre-trained embeddings
            print('\nUSING PRETRAINED EMBEDDINGS:\n')
            self.node_embeddings.weight.data = pretrained_embeddings

    def forward(self, node_indices):
        return self.node_embeddings(node_indices)


class GraphAutoencoder(nn.Module):
    def __init__(self, num_nodes, embedding_dim, latent_dim, edge_dim):
        super().__init__()

        # Node Embedding Layer:
        self.node_embeddings = NodeEmbeddingLayer(num_nodes, embedding_dim)

        # Encoder:

        # MLP for encoder - Creating weights (from edges) for NNConv
        self.edge_weight_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim * latent_dim) # Output weights for NNConv
        )
        # Compressing node embeddings
        self.encoder_conv = NNConv(
            in_channels=embedding_dim, # Input: Node embedding size
            out_channels=latent_dim,# Output: Compressed embedding size
            nn=self.edge_weight_encoder,       # MLP for edge features
            aggr='mean'         # Aggregate neighbor information
        )
        
        # MLP for decoder
        self.edge_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, edge_dim)
        )

    def encode(self, x, edge_index, edge_attr):
        return self.encoder_conv(x, edge_index, edge_attr)

    def decode(self, latent_x, edge_index):
        src, tgt = edge_index
        edge_input = torch.cat([latent_x[src], latent_x[tgt]], dim=1)
        return self.edge_predictor(edge_input)

    def forward(self, node_indices, edge_index, edge_attr):
        x = self.node_embeddings(node_indices)  # get node embeddings
        latent_x = self.encode(x, edge_index, edge_attr) # compress node embeddings, following edge weighting
        reconstructed_edge_attr = self.decode(latent_x, edge_index)
        return reconstructed_edge_attr


embedding_dim = 16  # length of vector representing each node embedding
latent_dim = 8      
edge_dim = train_graph_data.num_edge_features

model = GraphAutoencoder(global_num_nodes, embedding_dim, latent_dim, edge_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()

# Initialize the node embeddings with Node2Vec:
model.node_embeddings = NodeEmbeddingLayer(num_nodes=global_num_nodes, embedding_dim=embedding_dim, pretrained_embeddings=pretrained_node_embeddings).to(device)

# move everything to GPU (if available) before training
global_node_indices = global_node_indices.to(device)

edge_index = edge_index.to(device)
edge_attrs = edge_attrs.to(device)

test_edge_index = test_edge_index.to(device)
test_edge_attrs = test_edge_attrs.to(device)

train_graph_data = train_graph_data.to(device)
test_graph_data = test_graph_data.to(device)


# Training loop:
model.train()
for epoch in range(100):
    optimizer.zero_grad() # reset gradients

    # Forward pass
    reconstructed_edge_attr = model(global_node_indices, edge_index, edge_attrs)

    # Compute loss
    loss = loss_function(reconstructed_edge_attr, edge_attrs)

    # Backward pass and optimization
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Testing
model.eval()    # Set the model to evaluation mode

# Disable gradient tracking
with torch.no_grad():
    # Forward pass on the test graph
    test_reconstructed = model(global_node_indices, test_graph_data.edge_index, test_graph_data.edge_attr)
    
    # Compute the test loss
    test_loss = loss_function(test_reconstructed, test_graph_data.edge_attr)

# Print the test loss
print(f"Test Loss: {test_loss.item():.4f}")


edge_errors = torch.abs(test_graph_data.edge_attr - test_reconstructed).cpu().numpy()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# set threshold
threshold = np.percentile(edge_errors, 80)  # Set threshold
print("Selected Threshold:", threshold)


# Aggregate (e.g. mean) the error for each edge, reducing shape from (num_edges, num_features) to (num_edges,)
edge_errors_aggregated = edge_errors.mean(axis=1)

# Generate binary predictions: 1 = anomaly, 0 = normal
preds = (edge_errors_aggregated > threshold).astype(int)

# Convert true labels to numpy
true_labels = test_graph_data.y.values

print("True labels shape:", true_labels.shape)
print("Predictions shape:", preds.shape)

# 3. Confusion matrix and classification report
cm = confusion_matrix(true_labels, preds)
cr = classification_report(true_labels, preds)

print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)


# Reconstruction Error Plots:

# plot 1
plt.figure(figsize=(12, 8))
plt.hist(edge_errors, bins=200, alpha=0.7)
plt.axvline(x=threshold, color='red', linestyle='--', label="Threshold (80th Percentile)")
plt.xlabel("Reconstruction Error")  # Label the x-axis
plt.ylabel("Number of Edges")             # Label the y-axis
plt.title("Edge Reconstruction Error Distribution")  # Add a title
plt.legend()
# plt.show()

# plot 2
import numpy as np

# Separate normal and anomalous indices
normal_indices = np.where(edge_errors_aggregated <= threshold)[0]
anomalous_indices = np.where(edge_errors_aggregated > threshold)[0]

print(f"Normal edges: {len(normal_indices)}, Anomalous edges: {len(anomalous_indices)}")

from sklearn.utils import resample

# Sample size (e.g., 1% of the dataset)
sample_fraction = 0.01
normal_sample_size = int(len(normal_indices) * sample_fraction)
anomalous_sample_size = int(len(anomalous_indices) * sample_fraction)

# Stratified sampling
normal_sample = resample(normal_indices, n_samples=normal_sample_size, random_state=42, replace=False)
anomalous_sample = resample(anomalous_indices, n_samples=anomalous_sample_size, random_state=42, replace=False)

# Combine the sampled indices
sampled_indices = np.concatenate([normal_sample, anomalous_sample])
sampled_errors = edge_errors_aggregated[sampled_indices]

print(f"Sampled normal edges: {len(normal_sample)}, Sampled anomalous edges: {len(anomalous_sample)}")

plt.figure(figsize=(12, 8))
plt.scatter(sampled_indices, sampled_errors, alpha=0.6, label='Sampled Edges')
plt.axhline(y=threshold, color='red', linestyle='--', label="Anomaly Threshold")
plt.xlabel('Edge Index (1% Sample)')
plt.ylabel('Aggregated (Mean) Reconstruction Error')
plt.title('Scatter Plot of Stratified Sample of Reconstruction Errors')
plt.legend()
# plt.show()


# plot 3
# import collections

# # Count anomalies per node
# anomalous_edges = edge_errors_aggregated > threshold
# node_anomalies = collections.Counter()
# for i, is_anomalous in enumerate(anomalous_edges):
#     if is_anomalous:
#         src, tgt = edge_index[:, i]
#         node_anomalies[src.item()] += 1
#         node_anomalies[tgt.item()] += 1

# # Bar chart of anomalies per node
# nodes = list(node_anomalies.keys())
# counts = list(node_anomalies.values())

# plt.figure(figsize=(10, 6))
# plt.bar(nodes, counts, color='blue', alpha=0.7)
# plt.xlabel("Node")
# plt.ylabel("Number of Anomalous Edges")
# plt.title("Anomalies Per Node")
# plt.show()
