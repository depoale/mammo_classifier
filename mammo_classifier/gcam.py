""""GradCAM algorithm"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.cm as cm
from utils import create_new_dir


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names, output_path=None
):
    """Overlays gradCAM onto input image and saves it.
        Source. https://keras.io/examples/vision/grad_cam/
        ...
        Parameters
        ----------
        img_array: np.array
            input image
        model: keras.Model
        last_conv_layer_name: str
        classifier_layer_names: list
            list of layers after last_conv_layer_name
        output_path: str
            path where overlaid image will be saved"""
    
    img_array = img_array.reshape(1, 60, 60, 1)
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.input, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        #last_conv_layer_output_2 = model(img_array).get_layer
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
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
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255*heatmap)
    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    img_array*=255
    img_array = img_array.reshape(60, 60, 1)
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.3 + img_array
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    #Save the the superimposed image to the output path
    superimposed_img.save(output_path)

def get_gcam_images(images, best_model):
    """Creates and saves overlaid images using gradCAM calling make_gradcam_heatmap function."""
    # get list of layers after last_conv_layer_name
    classifier_layer_names = [layer.name for idx, layer in enumerate(best_model.layers) if idx > 8]
    create_new_dir('gCam')
    for i, image in enumerate(images):
        output_path = os.path.join('gCam', f'gCAM_{i}.png')
        make_gradcam_heatmap(image, model=best_model, last_conv_layer_name='conv_3', 
            classifier_layer_names=classifier_layer_names, output_path=output_path)
