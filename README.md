# Semantic Segmentation using variations of U-Net

## Description

In Computer Vision, Semantic Segmentation of any image means classifying 
each pixel on the image into a particular class (including an unknown class if required).

![alt text](https://miro.medium.com/max/882/0*oG83yRbn1PqM2rTY.png)

In the past years of Deep Learning, major advances have been made in Semantic Segmentation 
and a breakthrough occurred when Fully Convolutional Network (FCN) architectures like 
FCN-8 and U-Net were designed for an end to end masked-data-based learning. 
This article makes variations in the original U-Net architecture to 
increase its performance.


## Model Architectures

Original U-Net Architecture 

![alt text](https://miro.medium.com/max/882/1*lvXoKMHoPJMKpKK7keZMEA.png)


### 1. Standard U-Net

Standard U-Net architecture is similar to original U-Net with difference in sizes. Also, to avoid copy and crop and do 
pure concatenation, total symmetry is maintained and hence output of encoder block at a given depth 
can be directly concatenated with to its decoder counterpart. 

![alt text](readme_images/Model2.jpg)


### 2. U-Net TCED (Tightly Connected Encoder and Decoder)

![alt text](https://miro.medium.com/max/882/1*DBDnhVHCMLqbkb0XEBxq5Q.png)

The Copy and Crop layers of U-Net concatenate Spatial Information from the encoder 
block to the decoder block at the same depth. The concatenated layer is squeezed again
on the channel axis. These weights, copied directly from the encoder blocks, 
have spatial information that helps the decoder. 
The rationale for concatenating weights from the encoder at the same depth is to provide
network some relevant spatial information and adding on this, this approach provides
more amount of spatial context.

### 3. Res-U-Net

![alt text](https://miro.medium.com/max/1242/1*giPzCHu7C1Gw2xQtaQr4xQ.png)

Simple Encoder is replaced with Encoders with residual layers like in 
ResNet architecture. 
As U-Net may be very deep and ResNet helps in fending off 
degradation problem in deep architectures.

### Data set 
1. CARLA self-driving car dataset with 1060 images and mask.
2. KITTI SELF DRIVING CARS DATA

### Results

![alt text](https://miro.medium.com/max/750/1*QDbK90Ar21mg5hYZvYzxzA.png)

#### Inference on Carla Self Driving Car Dataset

![alt text](https://miro.medium.com/max/1400/1*f_bZokDyirlArd7m9Vamkw.png)

#### Inference on images clicked in San Francisco
![alt text](https://miro.medium.com/max/1400/1*uDr9aJosrB-zaemea8pRnQ.png)

https://miro.medium.com/max/1400/1*f_bZokDyirlArd7m9Vamkw.png
https://miro.medium.com/max/1400/1*uDr9aJosrB-zaemea8pRnQ.png


### Commands to run

#### Training
##### New Training Unet-TCED
` python .\src\driver.py -t training -n True -v UNetTCED`

##### New Training UNet STD
` python .\src\driver.py -t training -n True`

##### Training from previous checkpoint
`python .\src\driver.py  -t training`

##### Inference multiple images
`python .\src\driver.py  -t inference -m True -f .\data\carla\test\test_1\ -e png`

##### Inference single image
`python .\src\driver.py -t inference -i .\data\carla\test\test_1\7.png -d True -s True -e png`









