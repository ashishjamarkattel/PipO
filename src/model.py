import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

def model(size):

   
    base_model = ResNet50(

        weights="imagenet",
        include_top = False
        
    )

    base_model.trainable = False

    inputs = keras.layers.Input(shape=size)
    x = base_model(inputs,training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Flatten()(x)
    output = keras.layers.Dense(1)(x)
    model = Model(inputs=inputs,outputs=output)

    model.compile(optimizer=optimizer,loss=MeanSquaredError())

    return model


# if __name__ == "__main__":

#     size = (256,256,3) # size of input image.
#     resnet = model(size)