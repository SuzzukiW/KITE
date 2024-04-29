import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

class EpistasisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EpistasisDetector(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(EpistasisDetector, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    return correct / total

def run_epistasis_detector(X, y, hidden_dims, dropout_rate, learning_rate, batch_size, num_epochs, device):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = EpistasisDataset(X_train, y_train)
    test_dataset = EpistasisDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    output_dim = 1
    model = EpistasisDetector(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        train_accuracy = evaluate(model, train_loader, device)
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return model

class KITEDetector:
    def __init__(self, n_components=10, kernel='rbf', alpha=1.0, gamma=None):
        self.n_components = n_components
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
        self.krr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)

    def fit(self, X, y):
        self.X_kpca = self.kpca.fit_transform(X)
        self.krr.fit(self.X_kpca, y)

    def predict(self, X):
        X_kpca = self.kpca.transform(X)
        return self.krr.predict(X_kpca)

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

def compute_kernel_matrix(X, kernel='rbf', gamma=None):
    return pairwise_kernels(X, metric=kernel, gamma=gamma)

def preprocess_genotype_data(X):
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # One-hot encode categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X_imputed).toarray()

    return X_encoded