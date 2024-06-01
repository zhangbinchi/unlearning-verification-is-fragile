import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_data, choose_model, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('-data', '--dataset', default='mnist')
    parser.add_argument('-epoch', '--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('-lr', '--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('-fil', '--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('-m', '--model', default='mlp')
    parser.add_argument('-nl', '--mlp-layer', type=int, default=3,
                        help='number of layers of MLP (default: 3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-nc', '--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('-seed', '--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('-wd', '--wd', type=float, default=0.0005,
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('-uls', '--unlearn-size', type=int, default=10000,
                        help='number of unlearned samples (default: 100)')
    parser.add_argument('-N', '--N', type=int, default=10000,
                        help='number of data samples (default: 10000)')
    parser.add_argument('-M', '--M', type=int, default=50,
                        help='number of batch samples (default: 200)')
    parser.add_argument('-norm', '--norm', type=float, default=0.01,
                        help='norm of the perturbation (default: 0.01)')
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-t', '--t', type=int, default=5, help='number of repitition')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    torch.cuda.set_device(int(args.device_id))
    num_classes = 10
    args.num_classes = num_classes

    unlearn_acc_list = []
    retain_acc_list = []
    test_acc_list = []
    unlearn_f1_list = []
    retain_f1_list = []
    test_f1_list = []

    for seed in range(args.t):
        args.seed = seed
        model, _ = choose_model(args.dataset, args.model, args.mlp_layer, args.num_classes, args.filters)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        model = model.to(args.device)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        train_set, val_set, test_set = load_data(args.dataset, seed)

        if os.path.exists(f'pol/unlearn_set_{args.dataset}_{args.unlearn_size}_{args.seed}.npy'):
            unlearn_idx = np.load(f'pol/unlearn_set_{args.dataset}_{args.unlearn_size}_{args.seed}.npy')
        else:
            unlearn_idx = np.random.choice(len(train_set), args.unlearn_size, replace=False)
            np.save(f'pol/unlearn_set_{args.dataset}_{args.unlearn_size}_{args.seed}.npy', unlearn_idx)

        retain_idx = np.array(list(set(range(len(train_set))) - set(unlearn_idx)))
        unlearn_set = torch.utils.data.Subset(train_set, unlearn_idx)
        retain_set = torch.utils.data.Subset(train_set, retain_idx)
        assert len(train_set) == len(unlearn_set) + len(retain_set)

        batch_size = args.batch_size
        retain_size = len(retain_set)

        src_idx = torch.tensor(list(range(len(train_set))))
        trg = torch.zeros(len(train_set))
        trg[unlearn_idx] = 1

        init_model_path = f"pol/rand_retrain_misb_dist/{args.model}_init_seed_{args.seed}.pth"
        if os.path.exists(init_model_path):
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
        else:
            torch.save(model.state_dict(), init_model_path)

        best_epoch = 0
        lowest_val_loss = 1e10
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

        for epoch in range(args.epochs):
            train_loss = 0
            train_correct = 0
            train_total = 0
            k_batch = 0
            remain_idx = np.array(list(range(len(train_set))))
            retain_idx = np.array(list(set(remain_idx) - set(unlearn_idx)))
            while len(remain_idx) >= args.batch_size:
                batch_idx = np.random.choice(remain_idx, args.batch_size, replace=False)
                batch_data = trg[batch_idx]
                if torch.sum(batch_data) == 0:
                    remain_idx = np.array(list(set(remain_idx) - set(batch_idx)))
                    retain_idx = np.array(list(set(retain_idx) - set(batch_idx)))
                    model.train()
                    batch_train_data = [train_set[i][0] for i in batch_idx]
                    batch_train_labels = [train_set[i][1] for i in batch_idx]
                    inputs = torch.stack(batch_train_data).to(args.device)
                    targets = torch.tensor(batch_train_labels).to(args.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                else:
                    unl_batch_idx = batch_idx[np.where(batch_data == 1)]
                    if len(list(set(retain_idx) - set(batch_idx))) < torch.sum(batch_data):
                        print(
                            f"only {len(list(set(retain_idx) - set(batch_idx)))} data available but need to substitute {torch.sum(batch_data)} sample")
                        break
                    unl_data = [train_set[i][0] for i in unl_batch_idx]
                    unl_labels = [train_set[i][1] for i in unl_batch_idx]
                    inputs = torch.stack(unl_data).to(args.device)
                    targets = torch.tensor(unl_labels).to(args.device)

                    N = torch.sum(batch_data)

                    inputs, targets = inputs.to(args.device), targets.to(args.device)
                    model.eval()
                    with torch.autograd.set_grad_enabled(True):
                        # print(inputs.shape)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        gradients_unlearn = torch.autograd.grad(loss, model.parameters())

                    lowest_error = None
                    best_forge_batch = None
                    for k in range(args.M):
                        forge_batch = np.random.choice(retain_idx, int(N), replace=False)
                        batch_train_data = [train_set[b][0] for b in forge_batch]
                        batch_train_labels = [train_set[b][1] for b in forge_batch]
                        batch_train_data = torch.stack(batch_train_data).to(args.device)
                        batch_train_labels = torch.tensor(batch_train_labels).to(args.device)

                        with torch.autograd.set_grad_enabled(True):
                            outputs = model(batch_train_data)
                            loss = criterion(outputs, batch_train_labels)
                            gradients_forge = torch.autograd.grad(loss, model.parameters())
                        forge_error = sum([torch.norm(grad_forge - grad_unlearn).item() for grad_forge, grad_unlearn in
                                           zip(gradients_forge, gradients_unlearn)])

                        if lowest_error is None:
                            lowest_error = forge_error
                            best_forge_batch = forge_batch.copy()
                        elif forge_error <= lowest_error:
                            lowest_error = forge_error
                            best_forge_batch = forge_batch.copy()

                    model.train()
                    batch_idx[np.where(batch_data == 1)] = best_forge_batch

                    batch_data = [train_set[b][0] for b in batch_idx]
                    batch_labels = [train_set[b][1] for b in batch_idx]
                    batch_data = torch.stack(batch_data).to(args.device)
                    batch_labels = torch.tensor(batch_labels).to(args.device)
                    optimizer.zero_grad()
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    remain_idx = np.array(list(set(remain_idx) - set(batch_idx)))
                    remain_idx = np.array(list(set(remain_idx) - set(unl_batch_idx)))
                    retain_idx = np.array(list(set(retain_idx) - set(batch_idx)))

            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
            for data, targets in val_loader:
                data, targets = data.to(args.device), targets.to(args.device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

            val_loss /= len(val_loader)
            val_accuracy = 100.0 * val_correct / val_total

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(),
                           f"pol/rand_retrain_misb_dist/{args.dataset}_{args.model}_best_val_m_{args.M}_seed_{args.seed}.pth")

            unlearn_loader = torch.utils.data.DataLoader(unlearn_set, batch_size=args.batch_size, shuffle=False)
            retain_loader = torch.utils.data.DataLoader(retain_set, batch_size=args.batch_size, shuffle=False)
            print(f"Test/Unlearn/Retain accuracy for the retrained_misbehavior model in epoch {epoch}")
            test_acc, test_f1 = test(test_loader, model, args.device)
            unlearn_acc, unlearn_f1 = test(unlearn_loader, model, args.device)
            retain_acc, retain_f1 = test(retain_loader, model, args.device)

        print(f'Best model index for M = {args.M} is {best_epoch}')
        model.load_state_dict(
            torch.load(
                f"pol/rand_retrain_misb_dist/{args.dataset}_{args.model}_best_val_m_{args.M}_seed_{args.seed}.pth",
                map_location='cuda:0'))
        print(f"Test/Unlearn/Retain accuracy for the retrained_misbehavior model")
        test_acc, test_f1 = test(test_loader, model, args.device)
        unlearn_acc, unlearn_f1 = test(unlearn_loader, model, args.device)
        retain_acc, retain_f1 = test(retain_loader, model, args.device)
        unlearn_acc_list.append(unlearn_acc)
        retain_acc_list.append(retain_acc)
        test_acc_list.append(test_acc)
        unlearn_f1_list.append(unlearn_f1)
        retain_f1_list.append(retain_f1)
        test_f1_list.append(test_f1)
    print(f'{args.t} repitition AVG unlearn acc is {np.mean(unlearn_acc_list)}+{np.std(unlearn_acc_list)}')
    print(f'{args.t} repitition AVG retain acc is {np.mean(retain_acc_list)}+{np.std(retain_acc_list)}')
    print(f'{args.t} repitition AVG test acc is {np.mean(test_acc_list)}+{np.std(test_acc_list)}')
    print(f'{args.t} repitition AVG unlearn F1 is {np.mean(unlearn_f1_list)}+{np.std(unlearn_f1_list)}')
    print(f'{args.t} repitition AVG retain F1 is {np.mean(retain_f1_list)}+{np.std(retain_f1_list)}')
    print(f'{args.t} repitition AVG test F1 is {np.mean(test_f1_list)}+{np.std(test_f1_list)}')
