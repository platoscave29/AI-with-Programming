# PURPOSE: Create allows the user to build, train and validate a model. Descriptions of the arguments can be found 
#          below.
# Imports python modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import numpy as np
from workspace_utils import active_session

def build_model(arch, hidden_units, learning_rate, gpu):
    #Load the pre-trained model. 
    # Allow to choose between two different architectures
    if arch == "vgg16":
        model = models.vgg16(pretrained = True)
    elif arch == "densenet":
        model = models.densenet121(pretrained = True)
    else:
        print('Two available architecures are vgg16 or densenet. Default is vgg16.')
        model = models.vgg16(pretrained = True)
    
    # Build and train the classifer
    # Define a new, untrained feed-forward network as a classifier using ReLU activations and dropout
    # Train the classifier layers using backpropagations using the pre-trained network to get the features
    # Track the loss and accuracy on the validation set to determine the best hyperparameters
           
    # Freeze the parameters so we don't backprop through them

    for param in model.parameters():
        param.requires_grad = False
    
    # Allow the automatic use of CUDA if available

    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    # Building the classifer model
    # Input to model is 25088 and the output is 4096. 
    # For our problem modifying the output to be 102
    # or the number of flower categories
    if hidden_units > 4096:
        hidden_units = 512
        print('Hidden units greater than upstream layer, changed hidden units to default value')
    #elif hidden_units > 256:
        #hidden_units = 512
        #print('Hidden units greater than upstream layer, changed hidden units to default value')
              
              
    model.classifier = nn.Sequential(nn.Linear(25088,4096),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(4096,hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(hidden_units,256),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(256,102),
                                     nn.LogSoftmax(dim=1))

    #Define the loss 
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    model.to(device);
    
    print(f'Model is built')
    
    return model, optimizer, criterion, device
           

           
              
        
    