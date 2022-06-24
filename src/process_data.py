import pandas as pd 
import numpy as np 
import os
import math

def process_images(path,training=False,validation=False):

    """
    Map the images with corresponding angles.

    Args:
        path (string): location of the txt file which contain information.
    """
    images_angle = []
    data = []

    with open(path,"r") as f:
        
        lines = f.readlines()
        for l in lines:
            image, angle = l.split()
            images_angle.append([image,float(angle)*(math.pi)/180])



    if training:
    
        data = images_angle[:int(len(images_angle)*0.4)]

    if validation:
        
        data = images_angle[int(len(images_angle)*0.995):]



    return np.array(data)
            

        

# if __name__ == "__main__":
    
#     img_path  = "driving_dataset/data.txt"
#     data = process_images(img_path,training=True)
#     print(data.shape)