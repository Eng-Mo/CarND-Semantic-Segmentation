# Road Semantic Segmentation
---
This project is an implementation of Fully Convolutional Network for road semantic segmentation by labeling the road pixels in Image.

---
## Project Objectives
1. Extracting pretrained VGG-16 feature encoder.
2. Implementing FCN-8 decoder
3. Data Augmentation
4. Train the model on Kitti Road dataset 
5. Apllying the model to test data
6. Applying the model to Video
---
[image1]: ./runs/Images_results/loss_plot.png
[image2]: ./runs/Images_results/combined.png

## Data Augmentation
two technique for data augmentation. the first one is to flip the training data set from left to right. the second one is brightness augmentation by converting image to HSV and then scaling up or down the V channel randomly with a factor of 0.25. This was implemented in the gen_batch_function() in helper.py.

## Network Architecture

A pre-trained VGG-16 network model was used a an encoder by extracting the input, keep probability, layer3, layer4, layer7. The model was converted to FCN-8 by adding decoder network as following:
1. 1x1 convolution layer from VDD's layer7
2. 1x1 convolution layer from VDD's layer4
3. 1x1 convolution layer from VDD's layer3
4. Upsampling 1x1 layer7 with kernel 4 and strid 2
5. skip layer for 1x1 layer4 and upsamled the layer above
6. upsampling 1x1 layer4 with kernel 4 and strid 2
7. skip layer for 1x1 layer3 and upsamled the layer above
8. upsamling above layer with kernel 16 and stride 8

### Hyperparameters
1. random-normal kernel initializer with standard deviation 0.01 in all convolutional and upsampling layer
2. L2 kernel regularizer with L2 0.001 in all convolutional and upsampling layer
3. Epoch=60
4. Batch size= 5
5. Keep probabilites = .4
6. Learning rate= .0001

### Optimizer
The loss function for the network is cross-entropy, and oprimizer used is Adam optimizer.

## Result
![title][image1]
![title][image2]
Here's a [link to my video result1](./result1.mp4) 
Here's a [link to my video result2](./result2.mp4) 
Here's a [link to my video result3](./result3.mp4) 

## Dependencies
* Python 3
* TensorFlow
* NumPy
* SciPy
* OpenCV

