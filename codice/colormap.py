"""in case we develop an algorithm to segment images"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import image_dataset_from_directory
import cv2
from scipy.io import loadmat
from keras import layers
import keras
from models import cnn_model
from utils import get_data
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))
) 
img_height = 60
img_width = 60
split = 0.3
path = 'total_data'

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


colormap = loadmat(
    "./human_colormap.mat"
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    print(predictions)
    predictions = np.squeeze(predictions)
    print(predictions)
    #we don't need this singe we only have two classes
    #predictions = np.argmax(predictions, axis=2)
    #print(predictions)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        #bool l=1 if idx=mask
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    print(image.shape)
    print(colored_mask.shape)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()


def plot_predictions(images, colormap, model):
    for image in images:
        prediction_mask = infer(image_tensor=image, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(image, prediction_colormap)
        plot_samples_matplotlib(
            [image, overlay, prediction_colormap], figsize=(60, 60)
        )

if __name__ == '__main__':
    #print( type(colormap))
    #model = cnn_model()
    #model.built = True
    #model.load_weights('weights.h5', by_name = True, skip_mismatch = True)
    model = keras.models.load_model("best_model")
    data = image_dataset_from_directory(
    path,
    color_mode='grayscale',
    image_size=(64, 64),
    batch_size=1)
    data.shuffle(42)
 
    inputs = np.concatenate(list(data.map(lambda x, y:x)))
    targets = np.concatenate(list(data.map(lambda x, y:y)))

    images=inputs[:4]
    for image in images:
        #print(image.shape)
        pass
    #plot_predictions(inputs[:4], colormap, model=model)
    #plot_predictions(inputs[:4], colormap, model=model)

    def DeeplabV3Plus(image_size, num_classes):
        model_input = keras.Input(shape=(image_size, image_size, 1))
        resnet50 = keras.applications.ResNet50(
            weights=None, include_top=False, input_tensor=model_input
        )
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = DilatedSpatialPyramidPooling(x)

        input_a = layers.UpSampling2D(
            size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)
    
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        print(input_b)
        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = convolution_block(x)
        x = convolution_block(x)
        x = layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]),
            interpolation="bilinear",
        )(x)
        model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
        return keras.Model(inputs=model_input, outputs=model_output)


    model = DeeplabV3Plus(image_size=64, num_classes=2)
    model.summary()

    """
    ## Training
    We train the model using sparse categorical crossentropy as the loss function, and
    Adam as the optimizer.
    """

    #loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = 'binary_crossentropy'
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=["accuracy"],
    )

    history = model.fit(inputs, targets, validation_split=0.25, epochs=25)

    plt.plot(history.history["loss"])
    plt.title("Training Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history.history["accuracy"])
    plt.title("Training Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history.history["val_loss"])
    plt.title("Validation Loss")
    plt.ylabel("val_loss")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history.history["val_accuracy"])
    plt.title("Validation Accuracy")
    plt.ylabel("val_accuracy")
    plt.xlabel("epoch")
    plt.show()


