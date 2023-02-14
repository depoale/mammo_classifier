# Classificator for microcalcification in mammographies
[![Documentation Status](https://readthedocs.org/projects/cmepda-prj/badge/?version=latest)](https://cmepda-prj.readthedocs.io/en/latest/?badge=latest)
![GitHub repo size](https://img.shields.io/github/repo-size/depoale/cmepda_prj) 
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/depoale/cmepda_prj/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/depoale/cmepda_prj/tree/main)     

The aim of this project is to build and train a Convolutional Neural Network (CNN) for a deep-learning based classification of normal breast tissue VS breast tissue cointaining microcalcification clusters in mammograms' small portions. The neural network will be developed using the tools provided by the \textit{Keras} module in Python programming language. <br> Microcalcifications are tiny bright spots with a diameter of few hundred microns, that are commonly found in mammography screenings, even though their presence might be masked by the highly heterogenous surrounding breast tissue. While isolated microcalcifications are generally benign, when they turn up grouped in clusters with suspicious morphology and distribution patterns, they are associated with the risk of malignancy. 

# Dataset
The dataset of mammograms' sections that we are going to process in order to train our CNN is available on the INFN website, and is downloadable at the following link: https://www.pi.infn.it/~boccali/DL_class_data.tar.gz. This dataset is made up of 797 images, 414 of which represent normal breast tissue (they are labelled with "0") while the remaining 383 show breast tissue presenting microcalcification masses (they are labelled with "1"). They all are 60x60 pixels images extracted from real mammograms and converted into 8-bit grayscale. Here are some examples:
<img src="images/random_images.png" width="800"> 

## Data augmentation
 Since medical datasets are usually *small* (a few hundred samples), oftentimes performing data augmentation procedures is essential. This should help preventing overfitting during the training process of the neural network, hence it may improve both generalization and regularization of the given model.
In our case data augmentation procedures are implemented using **ImageDataGenerator** by Keras. This tool applies a series of random transformations to the original images of the dataset, such as rotations, vertical and orizontal flips, contrast modifications, rescalings, zoomings, brightness modifications and many others. 
Here we  report some examples of "augmented" images generated with this procedure starting from the initial dataset. 
<img src="images/augmented_images.png" width="800">
In this project the user can choose whether to perform or not the data augmentation procedures (by default the will not be performed). If the user's choice is affirmative, a new "augmented" dataset will be automatically created (it will be much ampler than the original one) and the CNN will be trained on its images.

## Wavelet-based filtering
Wavelet-based filters are often used in medical imaging in order to enhance images' information content, which mostly means improving the visibility of features of interest. Different filters 
can be employed to realize image denoising or to make objects' edges more distinguishable (increasing image contrast). In this project we are going to process mammograms' portions showing microcalcifications, therefore the objects to be preserved are microcalcifications themselves (small bright spots), whereas the noise to be filtered out is the non-uniform image background (heterogeneous breast tissue). The steps to be followed are: loading the images to be processed, implementing a 2-D Wavelet decomposition, analyzing the high spatial frequency (HF) and low spatial frequency (LF) components and finally obtaining the filtered images using a 2-D Wavelet reconstruction. <br>
 Among the many Wavelet families available, the best performing ones in our case are $sym3$ and $haar$. The decomposition level is set to be 3 and so we can set thresholds for decomposition coefficients in terms of the standard deviations of their distributions. In this case we will set to zero the "low spatial frequency approximation information", while keeping only the "high spatial frequency details" that exceed a certain number of standard deviations (best performances are obtained within 2 stdev). <br>> Here you can find some examples of Wavelet-based filtering procedures implemented on mammograms' portions from the original images' dataset (both normal and troubling breast tissue): \newpage 
<img src="images/random_wavelet.png" width="800"> <br>
In this project the user can choose whether to perform or not a Wavelet-based filtering procedure (by default it won't be performed), which Wavelet Transforms family has to be used to realize the filters between $sym3$ and $haar$ (the default one is $sym3$) and which threshold has to be set for the decomposition coefficients in terms of the standard deviations of their distributions (the default value is 1.5 stdev, anyway you are recommended not to go over 2 stdev). If the user's choice is affirmative, a new "wavelet" dataset will be automatically created (starting from the original dataset or from the "augmented" one) and the CNN will be trained on its images.

## External dataset
At the end of the training phase, the final model for the CNN breast tissue's classifier will be tested on an external dataset of mammograms' portions, in order to assess its generalization capability and its overall performances. The 60x60 pixels mammography sections, that will be processed in this conclusive testing stage, have been obtained cutting some mammograms contained in the dataset downloadable from Mendeley Data website at the following link: https://data.mendeley.com/datasets/ywsbh3ndr8/2. <br>
Here you can find some examples of mammograms' portions from the external dataset, representing both sane breast tissue and microcalcifications clusters. 
<img src="images/new_dataset.png" width="800"> <br>

# Hypermodel
The deisgned hypermodel for the CNN classifier is made up of three
convolutional blocks (containing *Conv2D*, *BathcNormalization*, *MaxPooling2D* and *Dropout* Keras layers) and a final fully-connected block (containing *Dense* Keras layers). <br>
The user can choose the possible values for the 3 available hyperparameters: $dim$, the number of output filters in the first convolution (this number doubles after each $Conv2D$ layer), $depht$, the number of $Dense$ layers in the fully connected block at the end of the hypermodel, and the $rate$ for the $Dropout$ layers. <br>
We have to underline that these parameters influence the model’s complexity, its generalisation capability and its performances. <br>
The default hyperparameter space is contained in the following table:
| Hyperparameters |     Values    | 
| ----------------| ------------- |
| dim            |  15, 25       | 
| dropout rate    |  0, 0.05,0.1  | 
| depth           |  1,2,3,4,5        | 

Here you can find the scheme representing the architecture of the designed CNN's hypermodel.
<img src="images/hypermodel_schema.jpg" width="1000"> <br>

# Model selection and model assessment
The model selection and model assessment procedure is presented in the diagram below: given an hypermodel and an hyperparameters space, the best model is selected with an internal Hold-out (validation set = 25\% of development set). A K-fold cross-validation (K=5) procedure is chosen to evaluate each model’s performance.
<img src="images/model_sel_assess.jpeg" width="800"> <br>
Hyperparameters search is executed using the $BayesianOptimizator$ tool provided by *keras-tuner*, in order to find the most suitable hyperparameters values for each of the five folds. The fraction of the hyperparameters' space that has to be explored in this phase can be selected by the user (the default value for the searching fraction is 25%).


# Models' ensamble
At this point we are left with 5 trained models or "experts" (one for each fold),
so an ensemble learning strategy is implemented using $PyTorch$.
The response of the "committee" is going to be a weighted average
of the single predictions, so that each expert will contribute to the final decision only with his "best part". The weights of the models' ensamble are obviously trained to maximise
its accuracy and represent the reliability of each expert
among the committee.
Finally, the ensemble’s performance is tested on the external dataset of mammograms' portions that we have already introduced. In this particular case the test set will contain only images showing microcalcifications clusters (Label = 1 only). <br>
The advantages of implementing a models' ensamble learning strategy are related to the CNN's robustness, architectural fliexibility and dinamic control of complexity, whereas its disadvantages are strictly connected to the computational expensiveness of the algorithm and the possible redundancy. 

# Classes
In order to implement the workflow described so far, two costum-made classes were built: *Data* and *Model*. 
 *Data* class is used to handle and manage the datasets of mammography gray-scale images to be processed: it is called to perform data augmentation and Wavelet-based filtering procedures and contains the **get_random_images** method, a useful funcion which returns random images' patterns from one or both classes (Label = 0 and Label = 1) of a certain dataset. <br> 
 *Model* class is used to carry out the aforementioned models' training and ensamble strategy. It is provided with many methods, such as:  <ul>
<li> **tuner**, which performs the hyperparameters search in the hyperparameter space set by the user </li>
<li> **fold**, which performs K-fold (in our case K = 5) for cross validation </li>
<li> **get_predictions**, which returns each model's prediction for all the images' patterns (used as input for the ensemble model) </li>
<li> **get_ensemble**, which trains and then saves the models' ensemble, the committee making final responses </li>
</ul>


# Performances
Using the default values for Wavelet-filtering settings and model hyperparameters, the classificator's performances we obtained are represented in the following graphs: <ul>
<li> Learning curves: here are the diagrams representing the Training and Validation Accuracy and the Training and Validation Loss recorded for one of the five total folds:
<img src="images/Fold_1.png" width="1000"> <br>
</li>
<li> ROC (Receiver Operating Characteristic) curves and AUC (Area Under the Curve): ROC curves are graphs which show the performance of a classification model at all classification thresholds, while the AUC measures the entire two-dimensional area underneath the entire ROC curve. In a ROC curve we plot the True Positive Rate (TPR) VS the False Positive Rate (FPR) at different classification thresholds: lowering the classification threshold the classifier will rank more items as positive, thus increasing both False Positives and True Positives Rates. <br> Here are the ROC curves relative to the testing data for each of the five folds, toghetere with their *MEAN* ROC Curve and its standard deviation: 
<img src="images/ROC_-_Testing_new.png" width="1000"> <br>
</li>
<li> Confusion Matrices: also called Error Matrices, they are specific table layouts that allow a fast visualization of the performances of an algorithm used for statistical classification.  Each row of a Confusion Matrix represents the instances in an actual class, while each column represents the instances in a predicted class, so that we can easily figure out the True or False Positive and the True or False Negative cases. Here are the five Confusion Matrices obtained for each of the 5 folds: 
<img src="images/Confusion_Matrices_new.png" width="1000"> <br>
</li>
<li> Ensamble' learning curves: here are the diagrams represenitng the Training, Validation and (external) Test accuracy and the Training, Validation and (external) Test Loss recorded for the final ensamble of the 5 models: 


# GradCAM
 Selecting the most reliable model (among the 5 available trained models, one for each fold) and according to the ebsamble's weights, the GradCAM (Gradient-weighted Class Activation Mapping) algorithm was employed to highlight which regions of an input image are relevant in the decision making
process to predict whether it represents normal breast tissue or microcalcifications' clusters. <br> 
In this project the user can choose the number of images randomly extracted from the processed dataset to visualize passing through the GradCAM algorithm (the default images number is 6 and you cannot exceed 25). Anyway here are some examples of mammograms' portions visualised with GradCAM: 
<img src="images/gCAM.png" width="800"> <br>