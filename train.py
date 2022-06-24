import numpy as np 
import cv2 as cv 
import pandas as pd 
from process_data import process_images
import tensorflow as tf
from model import model
import keras

## defining the generators

class dataset:

    def __init__(self,image_paths,height,width):

        self.paths = image_paths
        self.h  = height
        self.w = width


    def __getitem__(self,i):
        
        image,angle =  self.paths[i]
        image = cv.imread("driving_dataset/"+image)
        image = cv.resize(image,(self.h,self.w))
        image = image/255 ## normalizing the image

        return image,angle

    def __len__(self):

        return len(self.paths)




## defining the dataloader

class DataLoader(tf.keras.utils.Sequence):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        image = []
        angles = []
        for j in range(start, stop):
            image.append(self.dataset[j][0])
            angles.append(float(self.dataset[j][1]))

        return np.array(image,dtype="float32"),np.array(angles,dtype="float32")

    def __len__(self):

        return len(self.dataset)//self.batch_size

        


## hyperparameter

batch_size = 64
height = 224
width = 224
dim = 3
epoch = 2

img_path  = "driving_dataset/data.txt"
train_paths = process_images(img_path,training=True)
val_paths = process_images(img_path,validation=True)
train_data = dataset(train_paths,height,width)
val_data = dataset(val_paths,height,width)
train_dataloader = DataLoader(train_data,batch_size) 
val_dataloader = DataLoader(val_data,batch_size)

# print("The shape of train data: ")
# print(type(train_dataloader[0][1][0]))
# print("The shape of validation data: ")
# print(val_dataloader[0][0].shape,val_dataloader[0][1].shape)


## passing in the model
size = (height,width,dim)
tf.keras.backend.clear_session()
model = model(size)
print(model.summary())

## callbacks 
my_callbacks = [
   
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5')
    
]

# model = keras.models.load_model("model.04-1001.76.h5")
# model.set_weights(weights)
model.fit(train_dataloader,epochs=epoch,validation_data= val_dataloader,callbacks=my_callbacks)
# print(model.summary())