# Detecting Cars From Satellite Images using RetinaNet
<p align="center">
  <img width="auto" height="auto" src="https://github.com/R-Stefano/Remote-Sensing-Analysis/blob/master/result.png">
</p>

## About

Nowadays, Google and Apple have access to a huge number of mobile phones. These devices provide information about where we are and what type of transportation we are using. Although, Google is already using these information to suggest alternative routes, a smart management of: traffic lights, bridges, railway crossings and road works could further decrease the congestion problem. 

An alternative source of information could be remote sensing images to get a real-time analysis of the number of cars on the street. 
So I thought that could be interesting to develop a system able to determine the number of cars on the street in order to help to relieve car congestion. I decided to start with something of relative simple to get an understanding of the tools that I can use and how to efficiently apply them. 

**This first phase of the project focuses on developing an algorithm able to localize cars from satellite images.**

The article about this project can be found [Here](http://demiledge.com/artificialIntelligence/carDetection.php)

## Dependencies
* Tensorflow
* Numpy
* opencv
* Object Detection API

```
sudo pip3 install tensorflow 
sudo pip3 install opencv-python
```

[How to setup the Tensorflow's object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Training Setup
If you want to re-train the model to try to get better results. Run this command
```
wget ftp://gdo152.ucllnl.org/cowc/datasets/ground_truth_sets/*
```
copy the files inside the folders: *Postdam, Selwyn, Toronto and Utah* into a folder called **dataset**. Get rid of all the files excpeted the satellite images and their positive annotated images.

Then, create a folder called **data** and run 
```
python generate-TFFormat.py
```
As the filename says, this command will create the **shard** files used for train and test and put into the **data** folder.

>> **Note:** The **data** folder must be inserted into *models/research/*

The next step is to download the [model's weights](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models) and its [configuration file](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).

While the *model's weights* are going to be stored into *models/research/*, the *configuration file* should be renamed as **model.config**, [properly modified](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md) and put into *models/research/data*.

Always in *models/research/data*, create a file called **labels.pbtxt** with the following content:
```
item {
	id: 1
	name: 'car'
}
```

Finally, train the model running:
```
python object_detection/model_main.py \
--pipeline_config_path=data/model.config \
--model_dir=myModel \
--num_train_steps=50000 \
--sample_1_of_n_eval_examples=1 \
--alsologtostderr
```
## Prediction
On the other hand, if you just want to analyze an image you can run
```
python predict.py
```
add flag for image to predict
