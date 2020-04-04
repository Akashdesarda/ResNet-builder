# ResNet builder
Build a Customized ResNet architecture with n desired depth or layers. Based on [ResNet v2](https://arxiv.org/abs/1603.05027). It supports Tensorflow 2.0

### Contents

 1. [Overview](#overview)
 2. [Architecture](#architecture)
 3. [Data Loader](#dataloader)
 4. [Example](#example)
 5. [Documentation](#documentation)
 6. [TODO](#todo)

### Overview
ResNet have solved one of the most important problem- vanishing/exploding gradient problem and enables us to go much much deeper in our network. *The principal focus or aim of this Repo is,*

 1. Build a ResNetV2 network of any desired depth 
 2. Support for latest Tensorflow version ie tf 2.xx

*It supports total four network architecture:*

 1.  Build a complete ResNetV2 model with n no of residual layers with Fully connected layers included
2. Add n no of ResNetV2 layers in your already created model
3. Fine Tune ResNetV2 with fully connected layer included
4. Fine Tune ResNetV2 with fully connected layer not included

### Architecture 
Majorly the architecture is based on **ResNetV2**: [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027), though I have customised it a bit to further give better results. Enhancements include adding a Dropout layer, Skip connection over three layers, etc. I benchmark both version and tested on Cifar10 and saw 5-10% better result. Refer this [example](model.png) architecture with depth 9 for visualization. 

### Data Loader
Support for two data configration:

1.  Data is in seprate directory for training and validation data
2.  Data is in common directory

> Note: The format of data should follow Imagenet format ie some_directory/class/*.jpeg

### Example 
Refer to [Tutorial notebook](Tutorial.ipynb) for example & API refrence. 

### Documentation
Refer [ResNet Builder docs](https://akashdesarda.github.io/ResNet-builder/index.html) for all API refrence and documentation. 

### TODO

 1. Universal API to execute any job
 2. Add more example
