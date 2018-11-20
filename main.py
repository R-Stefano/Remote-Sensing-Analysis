import argparse
import tensorflow as tf
import cv2
import os
import tensorflow.contrib.slim as slim
import numpy as np
from modelBuilder import Yolov3_tiny
from predictor import filterPreds
import trainer

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

num_classes=1
trainImages=""
testImages=""
classesName=""


parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-m", "--model", help="Select which model use [tiny/yolo]", default="tiny")
parser.add_argument("-g", "--generate", help="Preprocessing the input images [true/false]", default="false")
parser.add_argument("mode", default="predict", help="Select if predict or train the model")


if __name__=='__main__':
    args=parser.parse_args()

    #Select the model
    if (args.model=="tiny"):
        print("Creating tiny YOLOv3..")
        model=Yolov3_tiny(num_classes, imgsize, anchors)

    with tf.Session() as sess:
        #Loading the weights
        model.loadWeights(sess)

        if (args.mode=="train"):
            trainer.train(model, sess)        
        
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
        




