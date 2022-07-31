from src import (
    config,
    model_dispatcher
)
import keras

# def get_model(size):
#     """Model for training the dataset

#     Returns:
#         model: Training model
#     """
   
#     base_model = model_dispatcher.BASE_MODEL[config.C_BASE_MODEL]      ## Base model to use default if resnet
#     base_model.trainable = config.TRAINABLE
#     base_model.layers[-1].trainable = True                 ## To train base model or not defaut if False
#     inputs = keras.layers.Input(shape=size)                 ## size default is 64 
#     x = base_model(inputs)
#     # x = keras.layers.Flatten()(x)
#     x = keras.layers.GlobalAveragePooling2D()(x)
#     for d_node in config.LAYERS:                            ## list of number of layers added default = []
#         x = keras.layers.Dense(d_node, activation="relu")(x)
#         # x = keras.layers.Dropout(0.5)(x)
#     output = keras.layers.Dense(config.N_CLASS, activation="softmax")(x)            ## number of class to predict default = 0
#     model = keras.Model(inputs=inputs,outputs=output)
  
#     model.compile(optimizer=config.OPTIMIZERS[config.C_OPTIMIZER],
#                   loss= config.LOSS[config.C_LOSS_FUNCTION],
#                   metrics="accuracy")

#     return model

def get_model(size):
    inputs = keras.layers.Input(shape=size)
    x = keras.layers.Conv2D(128, 3,  activation="relu")(inputs)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(128, 3,  activation="relu")(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(64, 5,  activation="relu")(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(64, 5,  activation="relu")(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    # x = keras.layers.Conv2D(32, 7,  activation="relu")(x)
    # x = keras.layers.MaxPooling2D((2,2))(x)
    # x = keras.layers.Conv2D(32, 7,  activation="relu")(x)
    # x = keras.layers.MaxPooling2D((2,2))(x)
    # x = keras.layers.Conv2D(16, 7,  activation="relu")(x)
    # x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    output = keras.layers.Dense(config.N_CLASS, activation="softmax")(x)

    model = keras.Model(inputs=inputs,outputs=output)
  
    model.compile(optimizer=config.OPTIMIZERS[config.C_OPTIMIZER],
                  loss= config.LOSS[config.C_LOSS_FUNCTION],
                  metrics="accuracy")

    return model
