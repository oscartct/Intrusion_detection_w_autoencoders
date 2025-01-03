import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
data = pd.read_csv('chunks_normalized.csv') 
tensor_data = torch.tensor(data.values, dtype=torch.float32)

# Create a DataLoader for training
batch_size = 32
train_loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True)

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize the model, loss function, and optimizer
input_dim = tensor_data.shape[1]
hidden_dim = 64  # Adjust based on your dataset
model = Autoencoder(input_dim, hidden_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 40
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'autoencoder_model.pth')

# Identify anomalies (reconstruction error)
model.eval()
with torch.no_grad():
    reconstructed = model(tensor_data)
    reconstruction_error = torch.mean((tensor_data - reconstructed) ** 2, dim=1)
    threshold = reconstruction_error.mean() + 3 * reconstruction_error.std()  # Example threshold
    anomalies = reconstruction_error > threshold
    print("Anomalies detected:", anomalies.sum().item())