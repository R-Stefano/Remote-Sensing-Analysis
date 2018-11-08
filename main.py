import argparse
import tensorflow as tf
import helper
import cv2
import os
import tensorflow.contrib.slim as slim
import numpy as np
from modelBuilder import tiny_yolov3

imgHeight=500
imgWidth=500

parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("mode", help="train - will train the model\n")
parser.add_argument("-m", "--model", help="Select which model use: tiny or YOLO", default="tiny")

if __name__=='__main__':
    args=parser.parse_args()
    
    if len(os.listdir('trainingImages')) == 0:
        loadImage=cv2.imread("images/Columbus/Columbus_01.png")
        #Prepare input image
        helper.prepareInputImage(loadImage)
        print("Image processed, loading first image for testing")
    else:
        print("The image has been already processed, just loading test image")

    test_image=np.expand_dims(cv2.imread("trainingImages/image_0.png"), axis=0)

    print("Test image loaded")
    inputs=tf.placeholder(tf.float32, [None, imgHeight, imgWidth, 3], name="RGB_image")
    #inputs=tf.expand_dims(inputs, axis=0)
    #Select the model
    if (args.model=="tiny"):
        print("Creating tiny YOLOv3..")
        output=tiny_yolov3(inputs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        out=sess.run(output, feed_dict={inputs:test_image})

        print(out.shape)



