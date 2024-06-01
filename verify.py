import argparse
import numpy as np
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from utils import load_data, load_dataset, params_to_vec, vec_to_params, batch_grads_to_vec
from model import MLP, AllCNN, ResNet18, ResNet50
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--mlp-layer', type=int, default=3,
                        help='number of layers of MLP (default: 3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    # Set the random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    train_set, test_set = load_dataset(args.dataset)
    # num_classes = len(train_set.classes) if args.num_classes is None else args.num_classes
    num_classes = 10
    args.num_classes = num_classes

    num_batches = int(np.ceil(len(train_set) / args.batch_size))
    criterion = nn.CrossEntropyLoss()
    verification_error = dict()
    for i in tqdm(range(args.epochs)):
        verification_error_per_epoch = []
        for j in range(num_batches):
            forged_batch = np.load(f"pol/forged_data/{args.model}_epoch_{i}_batch_{j}_seed_{args.seed}.npy")
            batch_data = [train_set[i][0] for i in forged_batch]
            batch_labels = [train_set[i][1] for i in forged_batch]
            batch_data = torch.stack(batch_data).to(args.device)
            batch_labels = torch.tensor(batch_labels).to(args.device)
            
            # load starting checkpoint
            if i == 0 and j == 0:
                model = torch.load(f"pol/model/{args.model}_init_seed_{args.seed}.pth")
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
            elif j == 0:
                model = torch.load(f"pol/forged_model/{args.model}_epoch_{i-1}_batch_{int(np.ceil(len(train_set) / args.batch_size))-1}_seed_{args.seed}.pth")
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
                optimizer.load_state_dict(torch.load(f"pol/optimizer/{args.model}_epoch_{i-1}_batch_{int(np.ceil(len(train_set) / args.batch_size))-1}_seed_{args.seed}.pth"))
            else:
                model = torch.load(f"pol/forged_model/{args.model}_epoch_{i}_batch_{j-1}_seed_{args.seed}.pth")
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
                optimizer.load_state_dict(torch.load(f"pol/optimizer/{args.model}_epoch_{i}_batch_{j-1}_seed_{args.seed}.pth"))
                
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            verifier_update = nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()
            model_recorded = torch.load(f"pol/forged_model/{args.model}_epoch_{i}_batch_{j}_seed_{args.seed}.pth")
            recorded_update = nn.utils.parameters_to_vector(model_recorded.parameters()).detach().cpu().numpy()
            verification_error_per_epoch.append(np.linalg.norm(verifier_update-recorded_update))
        verification_error[i] = verification_error_per_epoch
    
    print(verification_error.keys())

    np.save(f'pol/verification_error_{args.dataset}.npy', verification_error)
    
    # verification_error_plot = []
    # for i in range(args.epochs):
    #     verification_error_plot.append(verification_error[i])
    # verification_error_plot = np.array(verification_error_plot)
    # # Sort the data points
    # verification_error_plot.sort()

    # # Define error thresholds (you can define this based on your data range)
    # error_thresholds = np.linspace(verification_error_plot.min(), verification_error_plot.max(), 100)

    # # Calculate percentages
    # percentages = [np.mean(verification_error_plot < threshold) * 100 for threshold in error_thresholds]

    # # Plotting
    # plt.plot(error_thresholds, percentages)
    # plt.xlabel('Verification Error Threshold')
    # plt.ylabel('Percentage of Error Below Threshold (%)')
    # plt.grid(True)
    # plt.savefig(f'pol/error_percentage_{args.dataset}.pdf')
