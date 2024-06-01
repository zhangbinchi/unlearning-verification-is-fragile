import argparse
import copy
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from utils import load_data, load_dataset, params_to_vec, vec_to_params, batch_grads_to_vec
from model import MLP, AllCNN, ResNet18, ResNet50
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
import pickle


def test(testloader, model, device):
    criterion = nn.CrossEntropyLoss()
    loss = 0
    correct = 0
    total = 0
    pred_test = []
    label_test = []
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_test.append(predicted)
            label_test.append(labels)
    pred_test = torch.cat(pred_test, 0)
    label_test = torch.cat(label_test, 0)
    f1 = f1_score(label_test.detach().cpu().numpy(), pred_test.detach().cpu().numpy(), average='micro')
    print(f"Test Loss: {loss / len(testloader):.4f}, Test Accuracy: {100.0 * correct / total:.2f}%, Test Micro F1: {100.0 * f1:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--mlp-layer', type=int, default=3,
                        help='number of layers of MLP (default: 3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--unlearn-size', type=int, default=1000,
                        help='number of unlearned samples (default: 1000)')
    parser.add_argument('--pr', type=float, default=0.001,
                        help='perturbation learning rate (default: 0.001)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.set_device(4)
    args.device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    # Set the random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    train_set, test_set = load_dataset(args.dataset)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    # num_classes = len(train_set.classes) if args.num_classes is None else args.num_classes
    num_classes = 10
    args.num_classes = num_classes

    if args.model == 'mlp':
        model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        model_last = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        model_start_verify = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        model_start_forge = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'cnn':
        model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        model_last = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        model_start_verify = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        model_start_forge = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        model_last = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        model_start_verify = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
        model_start_forge = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet50':
        model = ResNet50(num_classes=args.num_classes).to(args.device)
        model_last = ResNet50(num_classes=args.num_classes).to(args.device)
        model_start_verify = ResNet50(num_classes=args.num_classes).to(args.device)
        model_start_forge = ResNet50(num_classes=args.num_classes).to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_verify = optim.SGD(model_start_verify.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_last = optim.SGD(model_last.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_batches = int(np.ceil(len(train_set) / args.batch_size))

    if os.path.exists(f'pol/unlearn_set_{args.dataset}_{args.unlearn_size}_{args.seed}.npy'):
        unlearn_idx = np.load(f'pol/unlearn_set_{args.dataset}_{args.unlearn_size}_{args.seed}.npy')
    else:
        unlearn_idx = np.random.choice(len(train_set), args.unlearn_size, replace=False)
        np.save(f'pol/unlearn_set_{args.dataset}_{args.unlearn_size}_{args.seed}.npy', unlearn_idx)
    retain_idx = np.array(list(set(range(len(train_set)))-set(unlearn_idx)))
    unlearn_set = torch.utils.data.Subset(train_set, unlearn_idx)
    retain_set = torch.utils.data.Subset(train_set, retain_idx)

    # Open the file in binary read mode
    with open(f'pol/train_datadist_{args.dataset}.pkl', 'rb') as file:
        # Load the data from the file
        dist_dict = pickle.load(file)
    
    if args.dataset == 'svhn':
        target = train_set.labels
    else:
        target = np.array(train_set.targets)
    
    label_dict = dict()
    for i in range(num_classes):
        label_dict[i] = np.arange(target.shape[0])[target == i]

    model_start_forge.load_state_dict(model.state_dict())
    model_last.load_state_dict(model.state_dict())
    optimizer_last.load_state_dict(optimizer.state_dict())

    lowest_error = 1e10
    best_epoch = None
    verification_error = dict()

    for epoch in tqdm(range(args.epochs)):
        verification_error_per_epoch = []
        # Training phase
        model.train()  # Set the model in training mode

        train_loss = 0
        train_correct = 0
        train_total = 0
        shuffled_indices = torch.randperm(len(train_set))

        for idx in range(num_batches):
            # Get the indices for the current mini-batch
            start_idx = idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(train_set))
            batch_idx = shuffled_indices[start_idx:end_idx]

            # Extract data and labels using the batch_indices
            batch_data = [train_set[i][0] for i in batch_idx]
            batch_labels = [train_set[i][1] for i in batch_idx]
            
            # Convert to tensors
            batch_data = torch.stack(batch_data).to(args.device)
            batch_labels = torch.tensor(batch_labels).to(args.device)

            # Perform forward pass
            outputs = model(batch_data)
            
            # Compute loss
            loss = criterion(outputs, batch_labels)

            # Perform backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model_start_verify.load_state_dict(model_start_forge.state_dict())
            optimizer_verify.load_state_dict(optimizer_last.state_dict())

            if len(set(batch_idx) & set(unlearn_idx)) == 0:
                forged_batch = batch_idx
                model_start_forge.load_state_dict(model.state_dict())
                # Randomly choose a data sample
                index = np.random.randint(len(retain_set))

                data, target = retain_set[index]
                target = torch.tensor([target]).to(args.device)
                data = data.unsqueeze(0).to(args.device)

                output = model_start_forge(data)
                loss = criterion(output, target)
                loss.backward()

                optimizer_forge = optim.SGD(model_start_forge.parameters(), lr=args.pr)
                output_new = model_start_forge(data)
                loss_new = criterion(output_new, target)
                optimizer_forge.zero_grad()
                loss_new.backward()
                optimizer_forge.step()

            else:
                forged_batch = batch_idx.copy()
                for i in range(forged_batch.shape[0]):
                    if forged_batch[i] in unlearn_idx:
                        for j in label_dict[train_set[forged_batch[i]][1]][dist_dict[forged_batch[i]]]:
                            if j not in unlearn_idx:
                                forged_batch[i] = j
                                break
                
                model_start_forge.load_state_dict(model_last.state_dict())
                optimizer_forge = optim.SGD(model_start_forge.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                optimizer_forge.load_state_dict(optimizer_last.state_dict())

                batch_data = [train_set[i][0] for i in forged_batch]
                batch_labels = [train_set[i][1] for i in forged_batch]
                batch_data = torch.stack(batch_data).to(args.device)
                batch_labels = torch.tensor(batch_labels).to(args.device)

                outputs = model_start_forge(batch_data)
                loss = criterion(outputs, batch_labels)

                # Perform backward pass and optimization
                optimizer_forge.zero_grad()
                loss.backward()
                optimizer_forge.step()

            model_last.load_state_dict(model.state_dict())
            optimizer_last.load_state_dict(optimizer.state_dict())
                
            outputs = model_start_verify(batch_data)
            loss = criterion(outputs, batch_labels)
            optimizer_verify.zero_grad()
            loss.backward()
            optimizer_verify.step()
            verifier_update = nn.utils.parameters_to_vector(model_start_verify.parameters()).detach().cpu().numpy()
            recorded_update = nn.utils.parameters_to_vector(model_start_forge.parameters()).detach().cpu().numpy()
            verification_error_per_epoch.append(np.linalg.norm(verifier_update-recorded_update))

        verification_error[epoch] = verification_error_per_epoch

    print('Finished Training')

    np.save(f'pol/verification_error_{args.dataset}_unlearn_{args.unlearn_size}_batch_{args.batch_size}_lr_{args.lr}.npy', verification_error)
    
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
    # plt.savefig(f'pol/error_percentage_{args.dataset}_unlearn_{args.unlearn_size}_batch_{args.batch_size}.pdf')
