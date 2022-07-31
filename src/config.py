import tensorflow as tf

## Directory location 
TRAIN_DIR = "I:\projects\PipO\dataset\Datos\Training-validation"
VALID_DIR = "I:\projects\PipO\dataset\Datos\Testing"
TEST_DIR = "I:\projects\PipO\dataset\Toy_val"


## model parameters
TRAINABLE = False
N_CLASS = 2             ## number of class
C_BASE_MODEL = "resnet"   ## base model
C_OPTIMIZER = "adam"      ## base optimizer
C_LOSS_FUNCTION = "cross_entropy"    # base loss function
LAYERS = []
HEIGHT = 128
WIDTH = 128


## hyper parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 20


## LOSS FUNCTION 
LOSS = {

        "cross_entropy": tf.keras.losses.CategoricalCrossentropy()
}

## OPTIMIZERS
OPTIMIZERS = {
    "adam": tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
}