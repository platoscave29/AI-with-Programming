# PURPOSE: Create a function that retrieves the following command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. Descriptions of the arguments can be found 
#          below. 

# Imports python modules
import argparse

def get_input_args_predict():
     
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    
    # Basic usage: python predict.py /path/to/image checkpoint
    parser.add_argument('image_path', action = 'store',
                        default = 'flowers/test/1/image_06754.jpg',
                        help = ' Path to image')
    
    # Checkpoint to load pre-trained model
    parser.add_argument('checkpoint',type = str, default = 'checkpoint.pth', 
                        help = 'Trained Model Checkpoint to Load')
    
    #Option to return the top K most likely cases
    parser.add_argument('--top_k', type = int, default = 5,
                        help = 'Specify the desired top K most likely classes')
        
    # Use mapping of categories to real names:
    parser.add_argument('--category_names',type= str, default = 'cat_to_name.json',
                        help = 'File name')
                        
    # Use GPU for training: python train.py data_dir --gpu
    parser.add_argument('--gpu', type = bool, default = 'False',
                        help = 'Type True to use the GPU, False to use the CPU')
    
    print(f"Inputs stored.")
                        
    return parser.parse_args()




