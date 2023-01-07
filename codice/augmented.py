"""data augmentation procedure.
    lowkey shitty bc test(& often val) accuracy fixed around 0.49 :(
    tried tweaking some param in ImageDataGenerator but nothing major happened"""

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from prova import get_data, get_model, plot 
from prova import callbacks,train_path, test_path, img_height, img_width, split, batch_size
train, val, test = get_data()

train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.18,
        horizontal_flip=True,
        vertical_flip=True,
        #fill_mode='reflect', #  nearest?
        validation_split=split)
        
train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    color_mode='grayscale', 
    class_mode='binary',
    subset='training')

val_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    class_mode='binary',
    subset='validation')

print(train_gen.next()[0].shape)
plt.imshow(train_gen.next()[0][1].squeeze(), cmap='gray')
plt.show()

model = get_model()
history = model.fit(train_gen, batch_size=batch_size , epochs=500, validation_data=val_gen, callbacks=callbacks)
plot(history=history)
print(f'test accuracy: {round(model.evaluate(test)[1],3)}')