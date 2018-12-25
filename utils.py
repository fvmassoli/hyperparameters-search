##################
# Pytorch imports
##################
import torch
import torch.nn as nn
import torchvision.transforms as t
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import WeightedRandomSampler


def make_weights_for_balanced_classes(images, n_classes):
    """
    Since the classes are not balanced let's create weights for them

    """
    count = [0] * n_classes
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * n_classes
    N = float(sum(count))
    for i in range(n_classes):
        weight_per_class[i] = N / float(count[i])
    weights = [0] * len(images)
    for idx, val in enumerate(images):
        weights[idx] = weight_per_class[val[1]]
    return weights


def get_transforms(train, hf=0.5, rr_degrees=30, gs=0.3,
                   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if train:
        return t.Compose([
            t.Resize(256),
            t.RandomHorizontalFlip(p=hf),
            t.RandomRotation(degrees=rr_degrees),
            t.RandomGrayscale(p=gs),
            t.RandomCrop(224),
            t.ToTensor(),
            t.Normalize(mean=mean, std=std),
        ])
    else:
        return t.Compose([
            t.Resize(256),
            t.CenterCrop(224),
            t.ToTensor(),
            t.Normalize(mean=mean, std=std),
        ])


def get_loaders(train_batch_size, num_workers=1, data_folder=None, cuda_available=False):
    print("Loading data...")
    train_data_set = ImageFolder(root=data_folder+'/train', transform=get_transforms(train=True))
    valid_data_set = ImageFolder(root=data_folder+'/valid', transform=get_transforms(train=False))
    weights = make_weights_for_balanced_classes(train_data_set.imgs, len(train_data_set.classes))
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    train_data_loader = DataLoader(dataset=train_data_set,
                                   sampler=sampler,
                                   batch_size=train_batch_size,
                                   num_workers=num_workers,
                                   pin_memory=cuda_available)
    valid_data_loader = DataLoader(dataset=valid_data_set,
                                   batch_size=8,
                                   num_workers=num_workers,
                                   pin_memory=cuda_available)
    print("Data loaded!!!")
    return train_data_loader, valid_data_loader


def get_model():
    print("Loading model...")
    model = models.vgg16(pretrained=True)
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.classifier[6] = nn.Linear(in_features=4096, out_features=102, bias=True)
    nn.init.xavier_normal_(model.classifier[6].weight)
    model.classifier[6].bias.data.fill_(0)
    model.classifier[6].weight.requires_grad = True
    model.classifier[6].bias.requires_grad = True
    print("Model loaded!!!")
    return model


def launch_tensorboard(logdir):
    import os
    os.system('tensorboard --logdir=~/ray_results/' + logdir)
    return
