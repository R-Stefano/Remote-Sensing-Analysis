# Satellite Car Detection using YOLOv3

## About

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


