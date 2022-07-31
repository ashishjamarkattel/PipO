import numpy as np 
import cv2 as cv 
import pandas as pd 
import tensorflow as tf
from keras.utils import to_categorical
import imgaug.augmenters as iaa


 
## defining the generators

class dataset:

    def __init__(self,data_df,height,width):

        self.images_paths = data_df.images.values  ## getting the dataframe values
        self.h  = height ## expected height of the image
        self.w = width   ## expected width of the data
        self.labels = data_df.labels.values ## getting the dataframe labels
        self.labels = to_categorical(self.labels)
        # augmentators
        self.augmentators = [
            iaa.Fliplr(1),
            iaa.DirectedEdgeDetect(alpha=(0.8), direction=(1.0)),
            iaa.Emboss(alpha=1, strength=1),
            iaa.Sharpen(alpha=1, lightness=1)
        ]

    def __getitem__(self,i):  # generators
        
        image =  self.images_paths[i]  ## getting the images
        labels = self.labels[i] # getting the labels of image
        image = cv.imread(image)
        image = cv.resize(image,(self.h,self.w), interpolation=cv.INTER_AREA)

        ## apply augmentation
        rand_num = np.random.randint(len(self.augmentators)-1)
        aug = self.augmentators[rand_num]
        image = aug.augment_image(image)
        image = image/255 ## normalizing the image

        return image,labels

    def __len__(self):

        return len(self.images_paths)




## defining the dataloader

class DataLoader(tf.keras.utils.Sequence):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        images = []
        labels = []
        for j in range(start, stop):
            images.append(self.dataset[j][0])
            labels.append(self.dataset[j][1])

        return np.array(images), np.array(labels)

    def __len__(self):

        return len(self.dataset)//self.batch_size

        
