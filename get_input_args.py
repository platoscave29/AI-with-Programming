# PURPOSE: Create a function that retrieves the following command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. Descriptions of the arguments can be found 
#          below. 

# Imports python modules
import argparse

def get_input_args():
     
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    
    # Basic usage: python train.py data_directory
    parser.add_argument('data_directory', action = 'store',
                        default = 'flowers/',
                        help = ' Data directory to load training data')
    
    parser.add_argument('--data_dir',type = str, default = 'flowers/', 
                        help = 'path to the folder of flowers')
    
    #Set directory to save checkpoints: python train.py data_dir --save_dir               save_directory
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth',
                        help = 'Save training models')
        
    # Choose architecture: python train.py data_dir --arch "vgg13"
    parser.add_argument('--arch',type= str, default = 'vgg16',
                        help = 'path to the CNN Model Architecture. Use either vgg16 or densenet.')
    
    # Set hyperparameters: python train.py data_dir --learning_rate 0.01 --               hidden_units 512 --epochs 20
    parser.add_argument('--learning_rate', type= float, default = 0.01,
                        help = 'specify learning rate for model')
                        
    # Set hyperparameters for hidden_units
    parser.add_argument('--hidden_units', type = int, default = 512, 
                        help = 'specify the number of hidden units')
                        
    # Set hyperparameters for epochs
    parser.add_argument('--epochs', type = int, default = 20, 
                        help = 'specify the number of epoch')
         
    # Use GPU for training: python train.py data_dir --gpu
    parser.add_argument('--gpu', type = bool, default = 'False',
                        help = 'Type True to use the GPU, False to use the CPU')
    
    print(f"Inputs stored.")
                        
    return parser.parse_args()



