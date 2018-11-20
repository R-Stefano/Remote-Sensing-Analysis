import argparse
import tensorflow as tf
import helper
import cv2
import os
import tensorflow.contrib.slim as slim
import numpy as np
from modelBuilder import Yolov3_tiny
from predictor import filterPreds
from helper import creatingLabels

'''
The truth bounding box have all the same shape 50x50. So, I don't need that the network learns 
to detect different shapes. So in the loss function at least for the moment, I avoid to assign the loss
to only the bounding box prediction with higher IOU and I'm going to use only the ones for where 
the grid is.

So, in order to create the label, I need to assign the labels to the grid. Suppose I have a grid of SxS.

It's like with the bins, I have to squash the img dim into S so, simply x//factor and y//factor and then reshape
to -1 so that I have SXS,1 where 
'''
#HYPERPARAMETERS
imgsize=416
step=400
num_anchors_scale=3
iou_threshold=0.4
conf_threshold=0.5
batch_size=1

anchors=[[(10., 13.), (16., 30.), (33., 23.)],
         [(30., 61.), (62., 45.), (59., 119.)],
         [(116., 90.), (156., 198.), (373., 326.)]]

num_classes=80#1
trainImages=""
testImages=""
classesName=""

classesNames=[]
with open("coco.names", "r") as f:
    for line in f:
        classesNames.append(line.replace("\n", ""))

parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
#parser.add_argument("mode", help="train - will train the model\n")
parser.add_argument("-m", "--model", help="Select which model use [tiny/yolo]", default="tiny")
parser.add_argument("-g", "--generate", help="Preprocessing the input images [true/false]", default="false")
parser.add_argument("mode", default="predict", help="Select if predict or train the model")


if __name__=='__main__':
    args=parser.parse_args()

    '''
    #Take the satelllite images and create the paches
    if (args.generate=="true"):
        inputData, labels=helper.prepareData(imgsize, step, num_classes)    
    testImg="test1.png"
    inputData=cv2.imread("test_predictions/"+testImg)
    inputData=cv2.resize(inputData[:,:,::-1], (imgsize,imgsize))
    '''

    img=cv2.resize(cv2.imread("lab_2.png"), (imgsize,imgsize))
    labels=np.max(img, axis=-1)

    targets=np.argwhere(labels!=0)

    #create box size labels
    box_size=np.tile([50,50], (targets.shape[0],1))

    #create isobject label
    isObj=np.ones((targets.shape[0],1))

    #isobj, x, y , w, h
    #NOW USE EXAMPAND DIM FOR BATCH SIZE
    y_true=np.expand_dims(np.concatenate((isObj,targets,box_size), axis=-1), axis=0)

    readyLabels=creatingLabels(y_true,imgsize, [13,26], batch_size)


    inputData=np.random.rand(imgsize,imgsize,3)*255

    inputs=tf.placeholder(tf.float32, [None, imgsize, imgsize, 3], name="RGB_image")
    
    #Select the model
    if (args.model=="tiny"):
        print("Creating tiny YOLOv3..")
        model=Yolov3_tiny(inputs, num_classes)

    with tf.Session() as sess:
        #Loading the weights
        model.loadWeights(sess)


        file=tf.summary.FileWriter("tensorboard/", sess.graph)

        preds,deb=sess.run([model.loss, model.deb1], feed_dict={inputs:np.expand_dims(inputData/255., axis=0), model.Y: readyLabels})
        
        print(deb[:10])
        if(args.mode=="predict"):
            results=filterPreds(preds,imgsize, conf_threshold, iou_threshold)

            for img in results:
                for key,value in img.items():
                    className=classesNames[key]
                    c=np.random.randint(0,255,3).tolist()
                    for v in value:
                        c1=v[0][:2]
                        c2=v[0][2:]
                        conf=v[1]

                        #BoundingBox
                        cv2.rectangle(inputData, tuple(c1), tuple(c2), c, conf)

                        #Textbackground
                        cv2.rectangle(inputData, (c1[0],int(c1[1]-15)),(int(c2[0]//2+c1[0]),c1[1]), c, -1)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text=className + " - " +str(format(conf, ".2f"))
                        cv2.putText(inputData,text,(c1[0],int(c1[1]-5)), font, 0.3,(0,0,0),1,cv2.LINE_AA)
                    
                cv2.imwrite("test_predictions/res_"+args.model+"_"+testImg, cv2.resize(inputData[:,:,::-1], (imgsize*2,imgsize*2)))
        




