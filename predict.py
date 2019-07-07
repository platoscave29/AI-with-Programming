# Objective is to predict a name along with the probability of that name. 
# A single image to be passed.

# Basic usage : python predict.py /path/to/image checkpoint
# Example: python predict.py 'flowers/test/102/image_08012.jpg' checkpoint.pth

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
from get_input_args_predict import get_input_args_predict
from image_preprocessing import image_preprocessing
from load_checkpoint import load_checkpoint

# Main program function defined below

def main():
# Get input arguments
# This function retrieves Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    
    in_arg = get_input_args_predict()
    
    # label mapping
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Allow the automatic use of CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() and in_arg.gpu else "cpu")
    if in_arg.gpu and torch.cuda.is_available():
        map_location = 'gpu'

    else:
        map_location = 'cpu'
    
    # Load the pretrained model
    model = load_checkpoint(in_arg.checkpoint, map_location, device)
      
    # Process Test Image
    image = image_preprocessing(in_arg.image_path)
    
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.type(torch.FloatTensor)
    image = image.to(device)
    
    model.to(device)   
    model.eval()
      
    # Calculate the class probabilities 
    with torch.no_grad():
        output = torch.exp(model.forward(image))
    
    #Output the probability and indices
    probs, indices = output.topk(in_arg.top_k)
    
    # Convert to class labels using class_to_idx
    # Make a dictionary of probabilities
    #probs = [float(prob) for prob in probs[0]]
    probs = probs.tolist()
    probs = probs[0]
     
    
    # Invert mapping from index to class
    inv_map = {v: k for k, v in model.class_to_idx.items()}
      
     
    #Make a dictionary of classes 

    classes = []
    #indices [0] is because indices is a tensor
    for index in indices[0]:
         if int(index) not in classes:
                classes.append(inv_map[int(index)])
        
   #np.vectorize(classes)
    
    classes_name = [cat_to_name[class_i] for class_i in classes]
    print(f"Class: {classes_name}.. "
          f"Probability: {probs}.. ")
    

# Call to main function to run the program
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)  
