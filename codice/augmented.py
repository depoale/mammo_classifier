from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from prova import get_data, get_model, train_path, test_path, img_height, img_width

train, val, test = get_data()

train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.18,
        horizontal_flip=True,
        #vertical_flip=True,
        fill_mode='reflect', #  nearest?
        validation_split=0.1)
        
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