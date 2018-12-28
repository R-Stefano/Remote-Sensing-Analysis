# Detecting Cars From Satellite Images using RetinaNet

## About
Nowadays, Google and Apple have access to a huge number of mobile phones. These devices provide information about where we are and what type of transportation we are using. Although, Google is already using these information to suggest alternative routes, a smart management of: traffic lights, bridges, railway crossings and road works could further decrease the congestion problem. 

An alternative source of information could be remote sensing images to get a real-time analysis of the number of cars on the street. 
So I thought that could be interesting to develop a system able to determine the number of cars on the street in order to help to relieve car congestion. I decided to start with something of relative simple to get an understanding of the tools that I can use and how to efficiently apply them. 

**This first phase of the project focuses on developing an algorithm able to localize cars from satellite images.**

The article about this project can be found [Here]()

## Dependencies
* Tensorflow
* Numpy
* opencv

```
sudo pip3 install tensorflow 
sudo pip3 install opencv-python
```

## Training Setup
If you want to re-train the model to try to get better results. Run this command
```
wget ftp://gdo152.ucllnl.org/cowc/datasets/ground_truth_sets/*
```
copy the files inside the folders: *Postdam, Selwyn, Toronto and Utah* into a folder called **dataset**. Get rid of all the files excpeted for the satellite images and their positive annotated images.

Then, create a folder called **data** and run 
```
python generate-TFFormat.py
```
As the filename says, this command will create the files used for train and test.


