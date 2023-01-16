"""data augmentation procedure.
    lowkey shitty bc test(& often val) accuracy fixed around 0.49 :(
    tried tweaking some param in ImageDataGenerator but nothing major happened
    try separating 0 from 1 in generating"""

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from utils import get_data, plot, callbacks, split
from models import get_model, get_bigger_model, cnn_model, cnn_classifier
from models import train_path, test_path, img_height, img_width
train, val, test = get_data(train_path='new_data/Train', test_path='new_data/Test')
train_path = 'new_data/Train'
batch_size=128
train_datagen = ImageDataGenerator(
        rotation_range=0,
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.18,
        horizontal_flip=True,
        vertical_flip=True,
        #fill_mode='reflect', #  nearest?
        validation_split=split)
        
train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=1,
    color_mode='grayscale', 
    class_mode='binary',
    #save_to_dir='augmented',
    subset='training')

val_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=1,
    color_mode='grayscale',
    class_mode='binary',
    subset='validation')


if __name__=='__main__': 
    model = cnn_classifier()
    history = model.fit(train_gen, batch_size=1 , epochs=100, validation_data=val_gen, callbacks=callbacks)
    plot(history=history)
    print(f'test accuracy: {round(model.evaluate(test)[1],3)}') 
    print(model.evaluate(test))