from torch.utils.data import random_split
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from model import MLP, AllCNN, ResNet18


def choose_model(dataset, modelname, mlp_layer, num_classes, filters):
    if modelname == 'mlp':
        if dataset == 'cifar10' or dataset == 'svhn':
            model_1 = MLP(num_layer=mlp_layer, num_classes=num_classes, filters_percentage=filters,
                          hidden_size=128, input_size=3072)
            model_2 = MLP(num_layer=mlp_layer, num_classes=num_classes, filters_percentage=filters,
                          hidden_size=128, input_size=3072)
        else:
            model_1 = MLP(num_layer=mlp_layer, num_classes=num_classes,
                          filters_percentage=filters, input_size=784)
            model_2 = MLP(num_layer=mlp_layer, num_classes=num_classes,
                          filters_percentage=filters, input_size=784)
    elif modelname == 'cnn':
        if dataset == 'mnist':
            model_1 = AllCNN(num_classes=num_classes, filters_percentage=filters, n_channels=1)
            model_2 = AllCNN(num_classes=num_classes, filters_percentage=filters, n_channels=1)
        elif dataset == 'cifar10' or dataset == 'svhn':
            model_1 = AllCNN(num_classes=num_classes, filters_percentage=filters, AvgPool=8)
            model_2 = AllCNN(num_classes=num_classes, filters_percentage=filters, AvgPool=8)
        else:
            model_1 = AllCNN(num_classes=num_classes, filters_percentage=filters)
            model_2 = AllCNN(num_classes=num_classes, filters_percentage=filters)
    elif modelname == 'resnet18':
        if dataset == 'mnist':
            model_1 = ResNet18(num_classes=num_classes, n_channels=1, filters_percentage=filters)
            model_2 = ResNet18(num_classes=num_classes, n_channels=1, filters_percentage=filters)
        else:
            model_1 = ResNet18(num_classes=num_classes, filters_percentage=filters)
            model_2 = ResNet18(num_classes=num_classes, filters_percentage=filters)

    return model_1, model_2


def load_dataset(dataset):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        raise ValueError('Undefined Dataset.')

    return train_set, test_set


def load_data(dataset, seed):
    train_set, test_set = load_dataset(dataset)
    torch.manual_seed(seed)

    if os.path.exists(f'data/{dataset}_split_indices_seed_{seed}.pth'):
        split_indices = torch.load(f'data/{dataset}_split_indices_seed_{seed}.pth')
        real_train_set = torch.utils.data.Subset(train_set, split_indices['train_indices'])
        val_set = torch.utils.data.Subset(train_set, split_indices['val_indices'])
    else:
        val_size = int(len(train_set) * 0.2)
        real_train_set, val_set = random_split(train_set, [len(train_set) - val_size, val_size],
                                               generator=torch.Generator().manual_seed(seed))
        torch.save({'train_indices': real_train_set.indices, 'val_indices': val_set.indices},
                   f'data/{dataset}_split_indices_seed_{seed}.pth')

    return real_train_set, val_set, test_set


def train(train_set, val_loader, model, lr, wd, num_epochs, bs, device, path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    best_val = 1e10
    for epoch in range(num_epochs):
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
        # Training phase
        model.train()  # Set the model in training mode

        train_loss = 0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)

        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()  # Set the model in evaluation mode
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total

        if val_loss < best_val:
            best_val = val_loss
            if path != "not_save":
                torch.save(model.state_dict(), path)

        # Print the validation loss and accuracy for each epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    print('Finished Training')


def test(testloader, model, device):
    criterion = nn.CrossEntropyLoss()
    loss = 0
    correct = 0
    total = 0
    pred_test = []
    label_test = []
    model.eval()
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
    f1 = f1_score(label_test.detach().cpu().numpy(), pred_test.detach().cpu().numpy(), average='macro')
    print(
        f"Test Loss: {loss / len(testloader):.4f}, Test Accuracy: {100.0 * correct / total:.2f}%, Test Micro F1: {100.0 * f1:.2f}%")
    return 100.0 * correct / total, 100.0 * f1
