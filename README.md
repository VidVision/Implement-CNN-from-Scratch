# Implement-CNN-from-Scratch

In this work, I have completed a simple Convolutional Neural Network architecture from scratch.
The goal is to apply a CNN Model on the CIFAR10 image data set and test the accuracy of the model on the basis of image classification.
CIFAR10 is a collection of images used to train Machine Learning and Computer Vision algorithms. It contains 60K images having dimension of 32x32 with ten different classes such as airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. We train our Neural Net Model specifically Convolutional Neural Net (CNN) on this data set.

CNN's are a class of Deep Learning Algorithms that can recognize and and classify particular features from images and are widely used for analyzing visual images. There are two main parts to a CNN architecture
- *  A convolution tool that separates and identifies the various features of the image for analysis in a process called as Feature Extraction.
- * A fully connected layer that utilizes the output from the convolution process and predicts the class of the image based on the features extracted in previous stages.

## Python and dependencies
Python 3 is used in this project.

- **environment.yaml**
Contains a list of needed libraries. 
$ conda env create -f environment.yaml

-**./train.py**
To run the model, simply run 
$ python train.py

## Data Loading ##
- **./data**
Download the CIFAR dataset with  provided script under ./data by:
$ cd data
$ sh get_data . sh
$ cd . . /
Microsoft Windows 10 Only
C: \ c o de  folder > cd data
C: \ c o de  folder \ data> get_data . bat
C: \ c o de  folder \ data> cd . .
The dataset is already downloaded so this step can be skipped.


-**./data/dataset_cifar.py**
This code reads the data batches in the data folder, and creates training, validation and testing X and Ys. 

## Model Implementation ##
-**./trainer.py**
The code optimizes the parameters of a model to minimize a loss function. We use training data X and y to compute the loss and gradients, and periodically check the accuracy on the validation set.

-**./modules/conv_classifier**
Ties each of the modules together to complete a CNN network. The network is constructed by a list of module definitions *in order* and handles both forward and backward communication between modules.
Typically, a convolutional neural network is composed of several different modules and these modules work together to make the network effective. For each module shown below, I have implemented a forward pass (computing forwarding results) and a backward
pass (computing gradients). These modules are the building block of *./modules/conv_classifier*.

1. **./modules/convolution.py**
The module for 2D Convolution, forward and backward implementation.

2. **./modules/relu.py**
The code for rectified linear units(ReLU) module, forward and backward implementation.

3. **./modules/max_pool.py**
The code for Max pooling module, forward and backward implementation.

4. **./modules/linear.py**
The code for Linear module, forward and backward implementation.

5. **./modules/softmax_ce.py**
The code for Softmax Cross Entropy module, forward and backward implementation. 
Computes softmax cross-entropy loss given the raw scores from the network.

## Optimizer ##
- **./optimizer**
* _base_optimizer.py*  : 
 Apply L2 penalty to the model. Update the gradient dictionary in the model
 
* sgd.py*
An optimizer is used to update weights of models. In practice, it is common to use a momentum term in SGD for better convergence. Specifically, we introduce a new velocity term vt and the update rule is as follows:

v<sub>t = βv<sub>t-1 -  η\frac{\partial L}{\partial w} \div \frac{\partial w}{\partial w}



