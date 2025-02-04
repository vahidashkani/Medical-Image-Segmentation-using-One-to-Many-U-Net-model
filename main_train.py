
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:12:14 2020

@author: vahid
"""

from dataset import DataGenerator
#from Fast_Unet2 import create_model
#from unet import create_model
from One_to_many_by_SPP_network import create_model
#from One_to_many_by_SPP_network import create_model
#from One_to_many_new import create_model
from keras import callbacks as cbks
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt

'''#gpu_id = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs", len(logical_gpus),"Logical GPUs")
  
    except RuntimeError as e:
        print(e)
'''

# Generators
training_generator = DataGenerator("/content/drive/MyDrive/Codes_HC18_Grand_Challenge_dataset/hc18_train.txt")
validation_generator = DataGenerator("/content/drive/MyDrive/Codes_HC18_Grand_Challenge_dataset/hc18_val.txt")
print(len(training_generator))
# create model
model = create_model(input_image_size=[256, 256, 1], n_labels=1)

batch_size = 1
callbacks = [cbks.ModelCheckpoint('/content/drive/MyDrive/Codes_HC18_Grand_Challenge_dataset/HK_augmented/proposed_model/333_weights.h5', monitor='val_loss', save_best_only=True),
            cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)]

epochs = 3
model.summary()
# train model
history = model.fit(
    x=training_generator, 
    steps_per_epoch=len(training_generator), 
    epochs=2, 
    callbacks=callbacks, 
    validation_data=validation_generator, 
    validation_steps=len(validation_generator), verbose=1, shuffle= True)

model.save("/content/drive/MyDrive/Codes_HC18_Grand_Challenge_dataset/HK_augmented/proposed_model/333.h5", include_optimizer=False)

# save logfile
loss_history = history.history
# a dictionary where the keys are column names (such as 'epoch', 'loss', 'val_loss', and 'lr').
log_df = pd.DataFrame.from_dict(loss_history)
log_df = log_df.drop(columns = ['lr'])
#log_df = log_df[['dice_coefficient','jaccard_coefficient','loss','val_dice_coefficient','val_jaccard_coefficient','val_loss']]
log_df = log_df[['dice_coefficient', 'jaccard_coef', 'loss', 'val_dice_coefficient', 'val_jaccard_coef', 'val_loss']]
log_df.to_csv('/content/drive/MyDrive/Codes_HC18_Grand_Challenge_dataset/HK_augmented/proposed_model/333_log_file.csv', sep=',', index = False)

#model.save("FL.h5", include_optimizer=False)

train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['dice_coefficient']
val_acc=history.history['val_dice_coefficient']
xc=range(epochs)

fig1 = plt.figure(1,figsize=(15,15))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print(plt.style.available)
plt.style.use(['classic'])
fig1.savefig("/content/drive/MyDrive/Codes_HC18_Grand_Challenge_dataset/figures/loss.png", dpi = fig1.dpi)


