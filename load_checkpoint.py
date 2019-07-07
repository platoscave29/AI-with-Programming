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
# inputs are the path to the checkpoint file
# the location where to store the data in terms of map_location (i.e. cpu or gpu)
# and also device location so we can run the model either on the gpu or cpu

def load_checkpoint(checkpoint_path, map_location, device):
   
    checkpoint = torch.load(checkpoint_path, map_location)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    return model