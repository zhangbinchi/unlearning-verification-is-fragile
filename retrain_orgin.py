import argparse
import numpy as np
import os
import torch
from utils import load_data, choose_model, test, train

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
    parser.add_argument('-t', '--t', type=int, default=1, help='number of repitition')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    torch.cuda.set_device(int(args.device_id))
    num_classes = 10
    args.num_classes = num_classes

    if not os.path.exists('pol'):
        os.mkdir('pol')

    retrained_unlearn_acc_list = []
    retrained_retain_acc_list = []
    retrained_test_acc_list = []
    retrained_unlearn_f1_list = []
    retrained_retain_f1_list = []
    retrained_test_f1_list = []

    ori_unlearn_acc_list = []
    ori_retain_acc_list = []
    ori_test_acc_list = []
    ori_unlearn_f1_list = []
    ori_retain_f1_list = []
    ori_test_f1_list = []

    for seed in range(args.t):
        args.seed = seed
        model, model_ori = choose_model(args.dataset, args.model, args.mlp_layer, args.num_classes, args.filters)
        model = model.to(args.device)
        model_ori = model_ori.to(args.device)
        torch.manual_seed(args.seed)
        np.random.seed(0)

        # Set the random seed for CUDA (if available)
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

        init_model_path = f"pol/rand_retrain_misb_dist/{args.model}_init_seed_{args.seed}.pth"
        if os.path.exists(init_model_path):
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
        else:
            torch.save(model.state_dict(), init_model_path)

        best_epoch = 0
        lowest_val_loss = 1e10
        unlearn_loader = torch.utils.data.DataLoader(unlearn_set, batch_size=args.batch_size, shuffle=False)
        retain_loader = torch.utils.data.DataLoader(retain_set, batch_size=args.batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        train(retain_set, val_loader, model, args.lr, args.wd, args.epochs, args.batch_size, args.device,
              f'pol/retrain/{args.dataset}_{args.model}_best_val_seed_{args.seed}.pth')
        train(train_set, val_loader, model_ori, args.lr, args.wd, args.epochs, args.batch_size, args.device,
              f'pol/origin/{args.dataset}_{args.model}_best_val_seed_{args.seed}.pth')

        test_acc, test_f1 = test(test_loader, model, args.device)
        unlearn_acc, unlearn_f1 = test(unlearn_loader, model, args.device)
        retain_acc, retain_f1 = test(retain_loader, model, args.device)
        retrained_unlearn_acc_list.append(unlearn_acc)
        retrained_retain_acc_list.append(retain_acc)
        retrained_test_acc_list.append(test_acc)
        retrained_unlearn_f1_list.append(unlearn_f1)
        retrained_retain_f1_list.append(retain_f1)
        retrained_test_f1_list.append(test_f1)

        ori_test_acc, ori_test_f1 = test(test_loader, model_ori, args.device)
        ori_unlearn_acc, ori_unlearn_f1 = test(unlearn_loader, model_ori, args.device)
        ori_retain_acc, ori_retain_f1 = test(retain_loader, model_ori, args.device)
        ori_unlearn_acc_list.append(ori_unlearn_acc)
        ori_retain_acc_list.append(ori_retain_acc)
        ori_test_acc_list.append(ori_test_acc)
        ori_unlearn_f1_list.append(ori_unlearn_f1)
        ori_retain_f1_list.append(ori_retain_f1)
        ori_test_f1_list.append(ori_test_f1)

    print(
        f'{args.t} repitition AVG retrain unlearn acc is {np.mean(retrained_unlearn_acc_list)}+{np.std(retrained_unlearn_acc_list)}')
    print(
        f'{args.t} repitition AVG retrain retain acc is {np.mean(retrained_retain_acc_list)}+{np.std(retrained_retain_acc_list)}')
    print(
        f'{args.t} repitition AVG retrain test acc is {np.mean(retrained_test_acc_list)}+{np.std(retrained_test_acc_list)}')
    print(
        f'{args.t} repitition AVG retrain unlearn F1 is {np.mean(retrained_unlearn_f1_list)}+{np.std(retrained_unlearn_f1_list)}')
    print(
        f'{args.t} repitition AVG retrain retain F1 is {np.mean(retrained_retain_f1_list)}+{np.std(retrained_retain_f1_list)}')
    print(
        f'{args.t} repitition AVG retrain test F1 is {np.mean(retrained_test_f1_list)}+{np.std(retrained_test_f1_list)}')

    print(
        f'{args.t} repitition AVG origin unlearn acc is {np.mean(ori_unlearn_acc_list)}+{np.std(ori_unlearn_acc_list)}')
    print(f'{args.t} repitition AVG origin retain acc is {np.mean(ori_retain_acc_list)}+{np.std(ori_retain_acc_list)}')
    print(f'{args.t} repitition AVG origin test acc is {np.mean(ori_test_acc_list)}+{np.std(ori_test_acc_list)}')
    print(f'{args.t} repitition AVG origin unlearn F1 is {np.mean(ori_unlearn_f1_list)}+{np.std(ori_unlearn_f1_list)}')
    print(f'{args.t} repitition AVG origin retain F1 is {np.mean(ori_retain_f1_list)}+{np.std(ori_retain_f1_list)}')
    print(f'{args.t} repitition AVG origin test F1 is {np.mean(ori_test_f1_list)}+{np.std(ori_test_f1_list)}')
