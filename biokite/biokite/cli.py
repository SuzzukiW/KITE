import argparse
from .detector import run_epistasis_detector, KITEDetector, preprocess_genotype_data
import torch
import numpy as np

def load_data(file_path):
    # Load data from a CSV file
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]  # Assume features are in all columns except the last one
    y = data[:, -1]   # Assume labels are in the last column
    return X, y

def main():
    parser = argparse.ArgumentParser(description='Biokite: Epistatic Interaction Detection')
    parser.add_argument('--data', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 32], help='Hidden layer dimensions')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n-components', type=int, default=10, help='Number of principal components')
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel function')
    parser.add_argument('--alpha', type=float, default=1.0, help='Regularization parameter')
    parser.add_argument('--gamma', type=float, default=None, help='Kernel coefficient')

    args = parser.parse_args()

    # Load and preprocess data
    X, y = load_data(args.data)
    X = preprocess_genotype_data(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run EpistasisDetector
    epistasis_model = run_epistasis_detector(X, y, args.hidden_dims, args.dropout_rate, args.learning_rate,
                                             args.batch_size, args.num_epochs, device)
    torch.save(epistasis_model.state_dict(), args.output)

    # Run KITEDetector
    kite_detector = KITEDetector(n_components=args.n_components, kernel=args.kernel,
                                 alpha=args.alpha, gamma=args.gamma)
    kite_detector.fit(X, y)
    torch.save(kite_detector, args.output + '_kite')

if __name__ == '__main__':
    main()