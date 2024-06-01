import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_data, load_dataset
from model import MLP, AllCNN, ResNet18, ResNet50
import argparse
import random
import os
import pickle
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
import pdb
from time import time


def worker_process(args, lr, epoch, idx, unlearn_idx, retain_set, train_set, seed, dist_dict, label_dict, device):

    # Load the model and optimizer states
    with open(f"pol/model/{args.model}_epoch_{epoch}_batch_{idx}_seed_{seed}.pth", 'rb') as model_file:
        model_forge = torch.load(model_file)
    with open(f"pol/data/{args.model}_epoch_{epoch}_batch_{idx}_seed_{seed}.npy", 'rb') as data_file:
        batch_idx = np.load(data_file, allow_pickle=True)
        
    criterion = nn.CrossEntropyLoss()

    if len(set(batch_idx) & set(unlearn_idx)) == 0:
        # Randomly choose a data sample
        index = np.random.randint(len(retain_set))

        data, target = retain_set[index]
        target = torch.tensor([target]).to(device)
        data = data.unsqueeze(0).to(device)
        
        output = model_forge(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer = optim.SGD(model_forge.parameters(), lr=lr)
        output_new = model_forge(data)
        loss_new = criterion(output_new, target)
        optimizer.zero_grad()
        loss_new.backward()
        optimizer.step()

        # Save the updated model
        torch.save(model_forge, f'pol/forged_model/{args.model}_epoch_{epoch}_batch_{idx}_seed_{seed}.pth')
        np.save(f"pol/forged_data/{args.model}_epoch_{epoch}_batch_{idx}_seed_{seed}.npy", batch_idx)

    else:
        forged_batch = batch_idx.copy()
        for i in range(forged_batch.shape[0]):
            if forged_batch[i] in unlearn_idx:
                for j in label_dict[train_set[forged_batch[i]][1]][dist_dict[forged_batch[i]]]:
                    if j not in unlearn_idx:
                        forged_batch[i] = j
                        break

        # load starting checkpoint
        if epoch == 0 and idx == 0:
            model_forge = torch.load(f"pol/model/{args.model}_init_seed_{seed}.pth")
            optimizer = optim.SGD(model_forge.parameters(), lr=args.lr)
        elif idx == 0:
            model_forge = torch.load(f"pol/model/{args.model}_epoch_{epoch-1}_batch_{int(np.ceil(len(train_set) / args.batch_size))-1}_seed_{seed}.pth")
            optimizer = optim.SGD(model_forge.parameters(), lr=args.lr)
            optimizer.load_state_dict(torch.load(f"pol/optimizer/{args.model}_epoch_{epoch-1}_batch_{int(np.ceil(len(train_set) / args.batch_size))-1}_seed_{seed}.pth"))
        else:
            model_forge = torch.load(f"pol/model/{args.model}_epoch_{epoch}_batch_{idx-1}_seed_{seed}.pth")
            optimizer = optim.SGD(model_forge.parameters(), lr=args.lr)
            optimizer.load_state_dict(torch.load(f"pol/optimizer/{args.model}_epoch_{epoch}_batch_{idx-1}_seed_{seed}.pth"))

        batch_data = [train_set[i][0] for i in forged_batch]
        batch_labels = [train_set[i][1] for i in forged_batch]
        batch_data = torch.stack(batch_data).to(device)
        batch_labels = torch.tensor(batch_labels).to(device)

        optimizer.zero_grad()
        outputs = model_forge(batch_data)
        loss = criterion(outputs, batch_labels)

        # Perform backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the updated model
        torch.save(model_forge, f'pol/forged_model/{args.model}_epoch_{epoch}_batch_{idx}_seed_{seed}.pth')
        np.save(f"pol/forged_data/{args.model}_epoch_{epoch}_batch_{idx}_seed_{seed}.npy", forged_batch)
        


if __name__ == '__main__':
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
    parser.add_argument('--num-process', type=int, default=1,
                        help='total number of processes (default: 1)')
    parser.add_argument('--process-id', type=int, default=0,
                        help='process id (range: 0 to num_process-1, default: 0)')
    args = parser.parse_args()
    args.device = torch.device("cuda")
    # args.device = torch.device("cpu")

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

    if not os.path.exists('pol/forged_model'):
        os.mkdir('pol/forged_model')
    if not os.path.exists('pol/forged_data'):
        os.mkdir('pol/forged_data')
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

    num_batches = int(np.ceil(len(train_set) / args.batch_size))

    for epoch in range(args.epochs):
        for idx in tqdm(range(args.process_id*int(np.ceil(num_batches/args.num_process)), min(num_batches, (args.process_id+1)*int(np.ceil(num_batches/args.num_process))))):
            worker_process(args, args.pr, epoch, idx, unlearn_idx, retain_set, train_set, args.seed, dist_dict, label_dict, args.device)
