import os
import sys
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling, Input
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, BinaryAccuracy
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))
) 
from keras.utils import image_dataset_from_directory

from PIL import Image

""" 
used to convert dataset to 'png' estension (supported by keras.utils.image_dataset_from_directory)   
for file in os.listdir('datasets/Mammography_micro/Test/0'):
    filename, extension  = os.path.splitext(file)
    if extension == ".pgm":
        new_file = "{}.png".format(filename)
        with Image.open(os.path.join('datasets/Mammography_micro/Test/0',file)) as im:
            im.save(new_file)
 """

batch_size = 12
img_height = 60
img_width = 60

train = image_dataset_from_directory(
  'data_png/Train',
  validation_split=0.3,
  subset="training",
  seed=123, color_mode='grayscale',
  image_size=(img_height, img_width),
  batch_size=batch_size)

val = image_dataset_from_directory(
  'data_png/Train',
  validation_split=0.3,
  subset="validation",
  seed=123,
  color_mode='grayscale',
  image_size=(img_height, img_width),
  batch_size=batch_size)

test = image_dataset_from_directory(
  'data_png/Test',
  color_mode='grayscale',
  image_size=(img_height, img_width),
  batch_size=batch_size)

#model
model = Sequential()
model.add(Input(shape=(60,60,1)))
#this layer normalizes the grayscale values from [0,255] to [0,1]
model.add(Rescaling(1./255))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(60,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(learning_rate=1e-4), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()



callbacks = (EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10, verbose=1))

start=time.time()
history = model.fit(train, batch_size=batch_size , epochs=500, validation_data=val, callbacks=callbacks)

""" fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper right") """

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc)+1)
#Train and validation accuracy 
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
#Train and validation loss 
plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
print(f'Elapsed time: {time.time()- start}')
plt.show()

print(f'test accuracy: {round(model.evaluate(test)[1],3)}')