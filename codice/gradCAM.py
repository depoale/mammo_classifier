import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from utils import read_imgs

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    grad_model.summary()
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:

        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)

        # Compute class predictions
        preds = grad_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


if __name__=='__main__':
    model = keras.models.load_model('prova_model.h5')
    model.summary()
    model.layers[-1].activation = None
    test_path = os.path.join(os.getcwd(),'data_png' ,'Test')
    #get three healty ex
    img_array, labels = read_imgs(test_path, classes=[1])
    print(labels.shape)
    print(img_array.shape)
    rnd_idx = np.random.randint(0, 100, size = 3)
    examples = img_array[rnd_idx]
    print(examples[0].shape)
    preds = model.predict(examples)
    print("Predicted:", (preds))
    heatmap = make_gradcam_heatmap(examples[0], model, last_conv_layer_name='conv_4')
    print(heatmap)
    heatmap = make_gradcam_heatmap(examples[1], model, last_conv_layer_name='conv_4')
    print(heatmap)
    heatmap = make_gradcam_heatmap(examples[2], model, last_conv_layer_name='conv_4')
    print(heatmap)
    #plt.matshow(heatmap)
    plt.show()