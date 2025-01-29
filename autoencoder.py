import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_normalize_data(file_path, train=True):
    """Load and preprocess data. If train=True, split into train and validation sets."""
    data = pd.read_csv(file_path)
    data = data.drop(columns=['start_time'], errors='ignore')  # Drop the 'start_time' column
    
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    if train:
        train_data, val_data = train_test_split(data_normalized, test_size=0.2, random_state=42)
        return t.tensor(train_data, dtype=t.float32), t.tensor(val_data, dtype=t.float32), scaler
    else:
        return t.tensor(data_normalized, dtype=t.float32)

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

def train_autoencoder(model, train_loader, val_loader, num_epochs, learning_rate):
    """Train the autoencoder model and track validation loss."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    training_losses, validation_losses = [], []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        
        # Compute validation loss
        model.eval()
        with t.no_grad():
            val_loss = sum(criterion(model(batch[0]), batch[0]).item() for batch in val_loader) / len(val_loader)
        validation_losses.append(val_loss)
        model.train()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    plot_loss_graph(training_losses, validation_losses)

def plot_loss_graph(training_losses, validation_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses, linestyle='-', color='b', label='Training')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, linestyle='-', color='r', label='Validation')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Autoencoder Training Loss Curve', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout(pad=2.0)
    plt.show()

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
    anomaly_indices = t.nonzero(anomalies).squeeze()

    print(f"Anomalies detected: {anomalies.sum().item()}")
    if anomalies.sum().item() > 0:
        print("Anomalies occurred at the following indices:")
        print(anomaly_indices.numpy())
    else:
        print("No anomalies detected.")
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(reconstruction_error.numpy(), label='Reconstruction Error', color='b')
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        plt.scatter(anomaly_indices.numpy(), reconstruction_error[anomaly_indices].numpy(), marker='o', color='r', label='Detected Anomalies', zorder=5)
        plt.xlabel('Time Segments', fontsize=16)
        plt.ylabel('Reconstruction Error', fontsize=16)
        plt.title('Reconstruction Error Across Data Segments', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout(pad=2.0)
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

    train_data, val_data, scaler = load_and_normalize_data(train_file, train=True)
    input_dim = train_data.shape[1]
    train_loader = create_dataloader(train_data, batch_size)
    val_loader = create_dataloader(val_data, batch_size, shuffle=False)

    if os.path.exists(model_file):
        model = load_model(model_file, input_dim, hidden_dim)
        print("Model loaded from file.")
    else:
        model = Autoencoder(input_dim, hidden_dim)
        train_autoencoder(model, train_loader, val_loader, num_epochs, learning_rate)
        save_model(model, model_file)

    test_data = load_and_normalize_data(test_file, train=False)
    detect_anomalies(model, test_data)
