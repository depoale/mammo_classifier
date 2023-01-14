"""data augmentation procedure.
    lowkey shitty bc test(& often val) accuracy fixed around 0.49 :(
    tried tweaking some param in ImageDataGenerator but nothing major happened
    try separating 0 from 1 in generating"""

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from models import get_data, get_model, plot, get_bigger_model, cnn_model
from models import callbacks,train_path, test_path, img_height, img_width, split, batch_size
from model_assessment import fold
train, val, test = get_data()
train_path = 'data_png/Train'
batch_size=64
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
    color_mode='grayscale', 
    class_mode='binary',
    #save_to_dir='augmented',
    subset='training')

val_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    class_mode='binary',
    subset='validation')


if __name__=='__main__': 
    model = cnn_model(verbose=True)
    #print(type(train_gen))
    #fold(data=(X,y), model=model)
    history = model.fit(train_gen, batch_size=batch_size , epochs=1000, validation_data=val_gen, callbacks=callbacks)
    plot(history=history)
    print(f'test accuracy: {round(model.evaluate(test)[1],3)}') 
    print(model.evaluate(test))