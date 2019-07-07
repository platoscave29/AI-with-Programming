# Preprocess the image prior to attempting to predict the image name

from PIL import Image
import numpy as np

def image_preprocessing(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
     # load the image
    pil_image = Image.open(image)
    width, height = pil_image.size
    
    # Scale the image witht the shortest side is 256 pixels while maintaining the aspect ratio
    # aspect ratio is the ratio of the width to height
    
    aspect_ratio = width/height
    
    # If width is greater than height, then the height should be 256
    if aspect_ratio > 1:
        new_width = round(aspect_ratio*width)
        pil_image.resize((new_width, 256))
    
    else:
        new_height = round(aspect_ratio*height)
        pil_image.resize((256, new_height))
            
    
    # crop out the center 224 of the image (dividing by 2 since we need the centerpoint)
    
    left_side = (pil_image.width-224)/2
    bottom_side = (pil_image.height-224)/2
    right_side = left_side + 224
    top_side = bottom_side + 224
    
    pil_image = pil_image.crop((left_side, bottom_side, right_side, top_side))
 
    #Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1
    np_image = np.array(pil_image)/255
       
    #Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])  
    np_image = (np_image - mean)/std
              
    # Reorder dimensions because Pytorch expects the color dimenstion to be 
    # the first dimension
    image = np_image.transpose(2,0,1)
    
    return image
    
   
    