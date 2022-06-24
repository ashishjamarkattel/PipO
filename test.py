import keras
import cv2 as cv
import numpy as np
from process_data import process_images
import sys
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm
import math
from model import model

# model_name = sys.argv[1]
# model = keras.models.load_model(model_name)

def testData(path,height,width):
    """
    Function iterate throught the test data path and predicts the angle

    Args:
        path (list(2d)): path to the test images
    """
    prediction = []
    original = []
    for loc in tqdm(path):

        image = cv.imread("driving_dataset/"+loc[0])
        image = cv.resize(image,(height,width)) ## resizie to given weigth and height
        image = np.expand_dims(image,axis=0) ## change in (batchsize,(size of image)) --> (1,224,244,3)
        pred = model.predict(image)
        prediction.append(pred[0][0])
        original.append(float(loc[1]))
        print(loc[0]," Original: ",loc[1],"Predicted : ",pred[0][0])
        

    plt.plot(original,color="blue")
    plt.plot(prediction,color="red")
    plt.show()


img_path  = "driving_dataset/data.txt"
height=  224
width = 224
test = process_images(img_path,validation=True)
testData(test,height,width)




