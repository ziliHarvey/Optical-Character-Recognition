# Optical Character Recognition and Feature Extraction
This repository contains implementation of optical character recognition and feature extraction  sytem using both MLP and CNN.

## Files included
**src/util.py** contains derivative of various activation functions  
**src/nn.py** contains functions for constructing a fully-connected neural network  
**src/fakeDataTest.py** training/testing with a single-layer fully-connected neural network on randomly generated dataset  
**src/realDataTest.py** training/testing with a single-layer fully-connected neural network on NIST36 dataset  
**src/findLetters.py**  function for extracting texts from neighboring pixels and plotting with bounding boxes  
**src/textExtraction.py** extracting hand-written texts from image and classified using nn trained in realDataTest.py  
**src/vaeCompression.py** compressing NIST36 dataset images with a 2-hidden-layer vanilla Autoencoder  
**src/pcaCompression.py** compressing NIST36 dataset images using principal component analysis approach  
**src/pytorchMLP.py** implementing a fc nn using Pytorch  
**src/pytorchCNN.py** implementing a CNN using Pytorch  
**src/cnnVisualization.py** visualizing feature maps and filters in a trained cnn  

