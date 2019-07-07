# Project: Image Classifier 
# train.py
# PROGRAMMER: Michelle Feole
# DATE CREATED: April 29, 2019                                 
# REVISED DATE: 
# PURPOSE: 
# Train a new network on a data set with train.py

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
#   Example call:
#    python train.py Data_Directory
##

# Imports python modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import numpy as np
from workspace_utils import active_session

#Used for Inference and Image Process
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
import sys


# Imports functions created for this program
from get_input_args import get_input_args
from process_data import process_data
from build_model import build_model

# Main program function defined below

def main():
    # This function retrieves Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args()
    
    # Load and preprocess the data
    trainloader, validloader, testloader, train_data, valid_data, test_data = process_data(in_arg.data_directory)
    
    # label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    # Building and training the model
    in_arg = get_input_args()
    
    # Build the model
    # Returns model, optimizer, criterion, and device
    model, optimizer, criterion, device = build_model(in_arg.arch, in_arg.hidden_units, in_arg.learning_rate, in_arg.gpu)
    
    # Train the model
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    print_every = 5

    from workspace_utils import active_session

    with active_session():

        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    # Prints the training loss, validation loss and validation accuracy
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(validloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(validloader):.3f}")

                    #reset running_loss to 0 and model back to training mode
                    running_loss = 0
                    model.train()   
    print('training completed')
    
    

    #saving the class indices
    model.class_to_idx = train_data.class_to_idx

    #creating the checkpoint
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'model': model,
                  'criterion': criterion,
                  'class_to_idx': model.class_to_idx,
                  'epoch': epochs,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, in_arg.save_dir)
    print(f'Model saved at specified checkpoint')
    

# Call to main function to run the program
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)  
    
    
    
    
    
    
    
