import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold

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

def main(args):
    # generate synthetic genotype and phenotype data with epistatic interactions
    X, y = make_classification(n_samples=args.num_samples, n_features=args.num_features, n_informative=args.num_informative,
                               n_redundant=args.num_redundant, n_repeated=args.num_repeated, n_classes=args.num_classes,
                               n_clusters_per_class=args.num_clusters_per_class, weights=None, flip_y=args.flip_y,
                               class_sep=args.class_sep, hypercube=args.hypercube, shift=args.shift,
                               scale=args.scale, shuffle=True, random_state=args.random_state)

    # set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the K-fold cross-validator
    kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_state)

    # perform K-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold+1}/{args.num_folds}")

        # split the data into training and testing sets for the current fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # create PyTorch datasets and data loaders for the current fold
        train_dataset = EpistasisDataset(X_train, y_train)
        test_dataset = EpistasisDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # initialize the model, loss function, and optimizer for the current fold
        input_dim = X_train.shape[1]
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
        output_dim = 1
        model = EpistasisDetector(input_dim, hidden_dims, output_dim, args.dropout_rate).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience, verbose=True)

        # initialize variables for early stopping
        best_accuracy = 0.0
        early_stop_counter = 0

        # training loop for the current fold
        for epoch in range(args.num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            train_accuracy = evaluate(model, train_loader, device)
            test_accuracy = evaluate(model, test_loader, device)
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

            # update the learning rate scheduler
            scheduler.step(test_accuracy)

            # check for early stopping
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Epistasis Detector')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--num_features', type=int, default=100, help='Number of features')
    parser.add_argument('--num_informative', type=int, default=10, help='Number of informative features')
    parser.add_argument('--num_redundant', type=int, default=0, help='Number of redundant features')
    parser.add_argument('--num_repeated', type=int, default=0, help='Number of repeated features')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--num_clusters_per_class', type=int, default=2, help='Number of clusters per class')
    parser.add_argument('--flip_y', type=float, default=0.01, help='Flip probability for labels')
    parser.add_argument('--class_sep', type=float, default=1.0, help='Class separation')
    parser.add_argument('--hypercube', action='store_true', help='Generate data in a hypercube')
    parser.add_argument('--shift', type=float, default=0.0, help='Shift for the data')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale for the data')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--hidden_dims', type=str, default='64,32', help='Comma-separated list of hidden layer dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for learning rate scheduler')
    parser.add_argument('--early_stop', type=int, default=20, help='Number of epochs for early stopping')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    main(args)