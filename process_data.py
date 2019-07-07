# Imports python modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import numpy as np
from workspace_utils import active_session

# Load, process, and transform data

def process_data(data_dir):
    
    #Data Directors for training, validation and testing
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Transforms for training, validation and testing set
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # 
    # Load the datasets with ImageFolder
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=250,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=250)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=250)
    
    print(f'Data loaded, transformed and processed.')
    
    return trainloader, validloader, testloader, train_data, valid_data, test_data