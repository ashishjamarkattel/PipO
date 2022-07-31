from src import config
import pandas as pd
import os
from tqdm import tqdm
tqdm.pandas()
from sklearn.preprocessing import LabelEncoder

def make_data(training=False, validation=False, testing=False):

    """create the dataframe for training or validation 
    i.e connects images with respective labels

    Args:
        training (bool, optional): Set true when data need is training. Defaults to False.
        validation (bool, optional): Set true when data need is of validation. Defaults to False.
    """
    print(" [INFO] CREATING THE DATAFRAME........ ")
    lb = LabelEncoder()
    data = pd.DataFrame()
    images = []
    labels = []
    if training:
        dir_loc = config.TRAIN_DIR
    elif validation:
        dir_loc = config.VALID_DIR
    elif testing:
        dir_loc = config.TEST_DIR
    else:
        return 

    try:
        for lbl in tqdm(os.listdir(dir_loc)):
            path = os.path.join(dir_loc, lbl)
            for train_img in os.listdir(path):
                image = os.path.join(path, train_img)
                images.append(image)
                labels.append(lbl)
                
        data["images"] = images
        data["labels_class"] = labels
        data["labels"] = lb.fit_transform(data["labels_class"])
    except:
        print("Something went wrong...")

    print(" [INFO] DATAFRAME CREATED........ ")


    return data.sample(frac=1)
    





    
