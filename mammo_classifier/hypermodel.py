"""Hypermodel builder and hps setter"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import GlorotNormal

img_height = 60
img_width = 60


def set_hyperp(hyperp_dict):
    """Hps dictionary is set to be a global variable so that it is accessible to the hypermodel as well.
    ...
    Parameters
    ----------
    hyperp_dict: dict
        hps dictionary"""
    global hyperp
    hyperp  = hyperp_dict

def get_search_space_size():
    """Calculate hyperparameters space size (based on the user-selected combinations)"""
    size = 1
    for key in hyperp:
            size*= len(hyperp[key])
    return size

def hyp_tuning_model(hp):
    """Hypermodel builder"""
    shape = (img_height, img_width, 1)
    model = Sequential()
    model.add(Input(shape=shape))
    hp_Dense_units = 256
    initializer = GlorotNormal()

    # set hps using the values stored in the dictionary hyperp
    hp_depth = hp.Choice('depth', hyperp['depth'])
    hp_Conv2D_init = hp.Choice('Conv2d_init', hyperp['Conv2d_init'])
    hp_dropout = hp.Choice('dropout', hyperp['dropout'])

    #three convolutional blocks
    model.add(Conv2D(hp_Conv2D_init, (3, 3), activation='relu', padding='same', strides=1, 
                     name='conv_1', input_shape=shape, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides = 2, name='maxpool_1'))
    model.add(Dropout(hp_dropout))

    model.add(Conv2D(2*hp_Conv2D_init, (3, 3), activation='relu', padding='same', 
                     strides=1, name='conv_2', kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2,  name='maxpool_2'))
    model.add(Dropout(hp_dropout))

    model.add(Conv2D(4*hp_Conv2D_init, (3, 3), activation='relu', padding='same', strides=1, 
                     name='conv_3', kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, name='maxpool_3'))
    model.add(Dropout(hp_dropout))

    model.add(Flatten())
    model.add(Dropout(hp_dropout + 0.1))

    #fully-connected block
    for i in range(hp_depth):
        units = hp_Dense_units/(i+1)
        model.add(Dense(units, activation='relu', name=f'dense_{i}', kernel_initializer=initializer))

    model.add(Dense(1, activation='sigmoid', name='output', kernel_initializer=initializer))

    model.compile(optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.9999), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
