# Classificator for microcalcification in mammographies
The aim of this project is to create and train a neural network able to distinguish between benign and malign microcalcification masses in mammographies.

## Dataset
Dataset source: https://www.pi.infn.it/~boccali/DL_class_data.tar.gz <br>
This dataset is made up of **797** images, **414** of which represent sane tissue and the remaining **383** diseased tissue.
<img src="images/random_images.png" width="500"> 

## Data augmentation
Since medical datasets are often *small* (a few hundred samples), oftentimes data augmentation procedure are performed. This should help preventing overfitting, hence it may improve both generalization and regularization of a given model. <br>
In this project, data augmentation is implemented using **ImageDataGenerator** by Keras. This tool applies a series of random transformations to the original images (e.g. rotation, vertical/orizontal flip, contrast modification...). <br>
Here are some examples of images generated with this procedure.
<img src="images/augmented_images.png" width="500"> 

## Wavelet
BOH!!

# Model selection and model assessment

