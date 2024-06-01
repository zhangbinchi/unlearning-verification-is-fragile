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


def train(train_set, test_loader, model, model_name, lr, wd, num_epochs, bs, seed, device, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    num_batches = int(np.ceil(len(train_set) / bs))

    if not os.path.exists('pol/data'):
        os.mkdir('pol/data')
    if not os.path.exists('pol/model'):
        os.mkdir('pol/model')
    if not os.path.exists('pol/optimizer'):
        os.mkdir('pol/optimizer')

    torch.save(model, f"pol/model/{model_name}_init_seed_{seed}.pth")

    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set the model in training mode

        train_loss = 0
        train_correct = 0
        train_total = 0
        shuffled_indices = torch.randperm(len(train_set))

        for batch_idx in tqdm(range(num_batches)):
            # Get the indices for the current mini-batch
            start_idx = batch_idx * bs
            end_idx = min(start_idx + bs, len(train_set))
            batch_indices = shuffled_indices[start_idx:end_idx]

            # Extract data and labels using the batch_indices
            batch_data = [train_set[i][0] for i in batch_indices]
            batch_labels = [train_set[i][1] for i in batch_indices]
            
            # Convert to tensors
            batch_data = torch.stack(batch_data).to(device)
            batch_labels = torch.tensor(batch_labels).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(batch_data)
            
            # Compute loss
            loss = criterion(outputs, batch_labels)

            # Perform backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)

            torch.save(model, f"pol/model/{args.model}_epoch_{epoch}_batch_{batch_idx}_seed_{seed}.pth")
            torch.save(optimizer.state_dict(), f"pol/optimizer/{args.model}_epoch_{epoch}_batch_{batch_idx}_seed_{seed}.pth")
            np.save(f"pol/data/{args.model}_epoch_{epoch}_batch_{batch_idx}_seed_{seed}.npy", batch_indices.numpy())
        
        train_loss /= num_batches
        train_accuracy = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()  # Set the model in evaluation mode
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
            
        val_loss /= len(test_loader)
        val_accuracy = 100.0 * val_correct / val_total
        
        # Print the validation loss and accuracy for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.2f}%")

    print('Finished Training')


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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
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

    if not os.path.exists('pol'):
        os.mkdir('pol')

    if args.model == 'mlp':
        model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'cnn':
        model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet50':
        model = ResNet50(num_classes=args.num_classes).to(args.device)

    train(train_set, test_loader, model, args.model, args.lr, args.weight_decay, args.epochs, args.batch_size, args.seed, args.device, args)
