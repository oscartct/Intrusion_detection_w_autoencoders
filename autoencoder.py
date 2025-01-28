import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

def load_and_normalize_data(file_path):
    """Load and preprocess test data"""
    test_data = pd.read_csv(file_path)
    test_data = test_data.drop(columns=['start_time'], errors='ignore')  # Drop the 'start_time' column
    scaler = StandardScaler()
    test_data_normalized = scaler.fit_transform(test_data)
    return t.tensor(test_data_normalized, dtype=t.float32)

def create_dataloader(data, batch_size, shuffle=True):
    """Create a DataLoader from tensor data."""
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, train_loader, num_epochs, learning_rate):
    """Train the autoencoder model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        for batch in train_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

def save_model(model, path):
    """Save the trained model to a file."""
    t.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path, input_dim, hidden_dim):
    """Load a saved autoencoder model."""
    model = Autoencoder(input_dim, hidden_dim)
    model.load_state_dict(t.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def detect_anomalies(model, data, threshold=None, plot=True):
    """Detect anomalies based on reconstruction error and plot the time segment vs reconstruction error."""
    with t.no_grad():
        reconstructed = model(data)
        reconstruction_error = t.mean((data - reconstructed) ** 2, dim=1)

    if threshold is None:
        threshold = reconstruction_error.mean() + 3 * reconstruction_error.std()

    anomalies = reconstruction_error > threshold
    anomaly_indices = t.nonzero(anomalies).squeeze()  # Get indices of anomalies

    print(f"Anomalies detected: {anomalies.sum().item()}")
    if anomalies.sum().item() > 0:
        print("Anomalies occurred at the following indices:")
        print(anomaly_indices.numpy())  # Print indices of detected anomalies
    else:
        print("No anomalies detected.")
    
    # Plot Time Segment vs Reconstruction Error
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(reconstruction_error.numpy(), label='Reconstruction Error', color='blue')
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        plt.scatter(anomaly_indices.numpy(), reconstruction_error[anomaly_indices].numpy(), color='red', label='Anomalies', zorder=5)
        plt.xlabel('Time Segments')
        plt.ylabel('Reconstruction Error')
        plt.title('Time Segment vs Reconstruction Error')
        plt.legend()
        plt.show()

    return reconstruction_error, threshold, anomalies

if __name__ == "__main__":
    train_file = 'train_data.csv'
    test_file = "test_data.csv"
    model_file = 'autoencoder_model.pth'
    batch_size = 32
    hidden_dim = 128
    num_epochs = 40
    learning_rate = 0.001

    train_data = load_and_normalize_data(train_file)
    input_dim = train_data.shape[1]
    train_loader = create_dataloader(train_data, batch_size)

    if os.path.exists(model_file):
        # Load 
        model = load_model(model_file, input_dim, hidden_dim)
        print("Model loaded from file.")
    else:
        # Train
        model = Autoencoder(input_dim, hidden_dim)
        train_autoencoder(model, train_loader, num_epochs, learning_rate)

        # Save 
        save_model(model, model_file)

    # Load and preprocess test data
    test_data = load_and_normalize_data(test_file)

    # Detect anomalies on the test data
    reconstruction_error, threshold, anomalies = detect_anomalies(model, test_data)
