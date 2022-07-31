import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf

BASE_MODEL = {
    "resnet": ResNet50(

        weights="imagenet",
        include_top = False
        
    )
}