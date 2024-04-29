import argparse
from .detector import run_epistasis_detector
import torch

def main():
    parser = argparse.ArgumentParser(description='Biokite: Epistatic Interaction Detection')
    parser.add_argument('--data', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 32], help='Hidden layer dimensions')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')

    args = parser.parse_args()

    # Load data from the input file
    # Assume the input file contains features (X) and labels (y)
    # You can modify this part based on your data format
    X, y = load_data(args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = run_epistasis_detector(X, y, args.hidden_dims, args.dropout_rate, args.learning_rate,
                                   args.batch_size, args.num_epochs, device)

    # Save the trained model
    torch.save(model.state_dict(), args.output)

if __name__ == '__main__':
    main()