# SSD-Object-Detection

I am working on a SSD Object Detector using fastai and pytorch 

This work was inspired from fastai's course Part -2 on 2018. https://www.youtube.com/watch?v=0frKXR-2PBY&list=PLfYUBJiXbdtTttBGq-u2zeY1OTjs5e-Ia&index=3&t=0s.
And the code was inspired from their courses repository on github. https://github.com/fastai/fastai/blob/master/courses/dl2/pascal-multi.ipynb 

I tried to increase the grids from just 4, 2 and 1 to 28, 14, 7, 4, 2, 1. I am trying to create a SSD Model which infers classes and bounding boxes from 6 different grid sizes.
In the original SSD paper they have used grid sizes of 19, 10, 7, 4, 2, 1. I have used a slightly different architecture. 

I have hooked outputs of the specified grid size from a pretrained model and pass those hooked outputs thorugh a ResBlock and a Out Conv layer which provides the classification  
and bounding box outputs.I also have a custom head to the pretrained model where I get inferences on grid sizes smaller than 7. 

The Aechitecture image is below:

![Architecture image](https://github.com/Samjoel3101/SSD-Object-Detection/blob/master/SSD%20Architecture%20Diagram.jpg)
