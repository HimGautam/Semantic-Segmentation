# Semantic-Segmentation
This project implements semantic segmentation by using deep learning.

![segmentation](https://user-images.githubusercontent.com/70597091/126811643-c9699dab-c24d-4864-97c8-dc11561e23af.gif)

# Overview

![Screenshot from 2021-07-23 21-53-51](https://user-images.githubusercontent.com/70597091/126812412-ea685c98-99c4-45f8-8b4e-e850a0545752.png)

The above picture shows the model used for this project. [EfficientNet B3](https://arxiv.org/pdf/1905.11946.pdf) is used as backbone for transfer learning which helps reduce the training time of the model.

![Screenshot from 2021-07-23 21-54-12](https://user-images.githubusercontent.com/70597091/126812424-d69c3299-8477-4f4f-81f0-44a491471f52.png)

The above picture shows the output of the network. The output will be of shape (B, H, W, C) where B is the Batch Size, H is Height of the image, W is the width of the image and C is the number of classes present in the segmentation ground truth. The output of the model is changed to segmentation map using ```argmax```. Then by using matplotlib.pyplot we can plot the segmentation map with disting color using command ```plt.imshow(model_output, cmap = "inferno")```.
