from src import (
    model, 
    model_dispatcher,
    dataLoader,
    config,
    process_data
)
import tensorflow as tf
from tqdm import tqdm
tqdm.pandas()

## get the dataframe
train_dataframe = process_data.make_data(training=True)
validation_dataframe = process_data.make_data(validation=True)

## get the dataloaders
train_dataloader = dataLoader.dataset(train_dataframe, config.HEIGHT, config.WIDTH)
validation_dataloader = dataLoader.dataset(validation_dataframe, config.HEIGHT, config.WIDTH)

## get the batch dataloaders
train_batch_data = dataLoader.DataLoader(train_dataloader, config.BATCH_SIZE)
validation_batch_data = dataLoader.DataLoader(validation_dataloader, config.BATCH_SIZE)


## passing in the model
tf.keras.backend.clear_session()
size = (config.HEIGHT, config.WIDTH, 3)
model = model.get_model(size)
print(model.summary())
## callbacks 
my_callbacks = [
   
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model/model.{epoch:02d}.h5',
        save_best_only=True,
        monitor="val_loss"
        ),
    
    tf.keras.callbacks.EarlyStopping(patience=3,
     min_delta=0.01, 
     monitor="val_loss")
    
]

model.fit(
    train_batch_data, 
    epochs=config.EPOCHS, 
    validation_data= validation_batch_data, 
    callbacks=my_callbacks
        )