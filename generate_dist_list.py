import pickle
import torch
import torch.nn.functional as F
from utils import load_dataset
import argparse

def calculate_distance(dataset):
    train_set, test_set = load_dataset(dataset)
    train_data = train_set.data
    if dataset == 'svhn':
        train_target = torch.tensor(train_set.labels)
    else:
        train_target = train_set.targets
    train_idx = list(range(train_data.shape[0]))
    train_data = torch.tensor(train_data).reshape(len(train_idx), -1)
    if dataset == 'cifar10':
        train_target = torch.tensor(train_target)
    print(train_data.shape,train_target.shape)
    feat_dist = {}
    for idx in set(train_idx):
        data_sample = torch.tensor(train_data[[idx]]).reshape(1,-1)
        same_class_data = train_data[train_target == train_target[idx]]
        similarity_vector = F.cosine_similarity(same_class_data.float(), data_sample.float(), dim=1) #size len(retain_idx)
        sort_idx = torch.argsort(similarity_vector, descending=True)
        feat_dist[idx] = sort_idx
    return feat_dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--dataset', default='mnist')
    args = parser.parse_args()

    dataset = 'cifar10'
    feat_dist = calculate_distance(args.dataset)
    with open(f'pol/train_datadist_{args.dataset}.pkl', 'wb') as file:
        pickle.dump(feat_dist, file)
