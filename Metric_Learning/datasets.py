import json

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets

from params import args


# 0: 猫, 1: 犬
def load_fname(json_path):
    with open(json_path) as f:
        d = f.read()
    data_dict = json.loads(d)

    image_fname_list = []
    for species in data_dict:
        # for seed in data_dict[species]:
        #     image_fname_list += data_dict[species][seed]['images']
        image_fname_list += data_dict[species]['images']
    return image_fname_list


def load_images(image_fname_list, transform):
    ret_images = []
    ret_labels = []
    for fname in image_fname_list:
        # path = './COVID_data/' + fname + '.jpg'
        # img = Image.open(path).convert('RGB')
        # img = transform(img)
        # ret_images.append(img)
        # if fname[:2] == 'CA':
        #     label = 0
        # elif fname[:2] == 'Co':
        #     label = 1
        # elif fname[:2] == 'HC':
        #     label = 2
        # elif fname[:2] == 'SP':
        #     label = 3

        path = './COVID_new/' + fname + '.jpg'
        img = Image.open(path).convert('RGB')
        img = transform(img)
        ret_images.append(img)
        if fname[:5] == 'COVID':
            label = 0
        elif fname[:5] == 'NORMA':
            label = 1
        elif fname[:5] == 'TUBER':
            label = 2

        ret_labels.append(label)

        # label = 1 if fname[0].islower() else 0
        # ret_labels.append(label)
    
    ret_images = torch.stack(ret_images)
    ret_labels = torch.Tensor(ret_labels)
    return ret_images, ret_labels


def load_datasets(train_json, test_json, transform):
    train_fnames = load_fname(train_json)
    fnames = load_fname(test_json)

    valid_size = int(len(fnames) * 0.3)
    valid_fnames = np.random.choice(fnames, valid_size)
    test_fnames = []
    for fname in fnames:
        if fname not in valid_fnames:
            test_fnames.append(fname)

    X_train, y_train = load_images(train_fnames, transform)
    X_valid, y_valid = load_images(valid_fnames, transform)
    X_test, y_test = load_images(test_fnames, transform)

    train_dataset = TripletSampler(X_train, y_train)
    valid_dataset = TripletSampler(X_valid, y_valid)
    test_dataset = TripletSampler(X_test, y_test)

    return train_dataset, valid_dataset, test_dataset


def load_test(test_json, transform):
    test_fnames = load_fname(test_json)
    X_test, y_test = load_images(test_fnames, transform)
    test_dataset = TripletSampler(X_test, y_test)
    return test_dataset


class TripletSampler(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)
 
    def __getitem__(self, idx):
        x = self.inputs[idx]
        t = self.targets[idx]
        xp_idx = np.random.choice(np.where(self.targets == t)[0])
        xn_idx = np.random.choice(np.where(self.targets != t)[0])
        xp = self.inputs[xp_idx]
        xn = self.inputs[xn_idx]
        return x, xp, xn, t