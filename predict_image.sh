#!/bin/sh
# For training the image classifier
#                                                                            
#
# Usage: sh predict_images.sh    -- will run program from commandline within Project Workspace

python predict.py 'flowers/test/102/image_08012.jpg' checkpoint.pth