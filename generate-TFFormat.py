import numpy as np
import os
import cv2
import tensorflow as tf
from object_detection.utils import dataset_util
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

def create_tf_example(filename, image, labels, imgsize):
    # TODO(user): Populate the following variables from your example.
    height = imgsize # Image height
    width = imgsize # Image width
    encoded_image_data = image # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for lb in labels:
        xmins.append(lb[1])
        ymins.append(lb[0])
        xmaxs.append(lb[3])
        ymaxs.append(lb[2])
        classes_text.append("car".encode('utf-8'))
        classes.append(int(lb[4]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    
    return tf_example

patchsize=500
stride=400
box_size=20

img_count=0
dataset_folder="dataset"
shard_train="data/train-dataset.record"
shard_test="data/test-dataset.record"
test_count=0
train_count=0

train_examples_shard=512
test_examples_shard=128
num_shards=10

with contextlib2.ExitStack() as tf_record_close_stack:
    train_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
      tf_record_close_stack, shard_train, num_shards)
    
    test_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
      tf_record_close_stack, shard_test, num_shards)
    #every image has composed by 2 files: image, positive
    for _,_,fileList in os.walk(dataset_folder):
        sortedFiles=sorted(fileList)
        for i in range(0,len(sortedFiles), 2):
            image=cv2.imread(dataset_folder+"/"+sortedFiles[i])
            #convert to rgb
            rgbImg=image[:,:,::-1]

            posAnn=cv2.imread(dataset_folder+"/"+sortedFiles[i+1])

            #Slide vertically
            for sH in range(0, image.shape[0], stride):
                eH=sH+patchsize
                #Slide horizontally
                for sW in range(0,image.shape[1], stride):
                    eW=sW+patchsize

                    #Collect the patch from the image
                    patch=rgbImg[sH:eH,sW:eW]

                    #If the patch is smaller than the patch size
                    if(patch.shape[0]<patchsize or patch.shape[1]<patchsize):
                        #Resize it
                        patch=cv2.resize(patch, (patchsize,patchsize))

                    #encode it
                    encodedImg=cv2.imencode('.png', patch)[1].tostring()

        
                    #Collect the same patch from the annotation image
                    patchPosAnnotated=posAnn[sH:eH,sW:eW]

                    #If the patch is smaller than the patch size
                    if(patch.shape[0]<patchsize or patch.shape[1]<patchsize):
                        #resize it
                        patchPosAnnotated=cv2.resize(patchPosAnnotated, (patchsize,patchsize))

                    #Create labels
                    labels=np.max(patchPosAnnotated, axis=-1)

                    #shape num_targets,2
                    targets=np.argwhere(labels!=0)

                    #The format that I need is ymin, xmin, ymax, xmax normalized and the class starting from 0 and class name
                    targets=np.concatenate((targets-box_size/2, targets+box_size/2), axis=-1)

                    #create label class 0
                    isObj=np.zeros((targets.shape[0],1)).astype(int)
                    
                    targets=np.concatenate((targets, isObj), axis=-1)

                    #normalize boxes
                    targets[:,:4]=targets[:,:4]/patchsize

                    if (np.random.rand()<0.2 and test_count<num_shards*test_examples_shard):
                        imageName="img_"+str(test_count)+".png"
                        tf_example = create_tf_example(imageName, encodedImg, targets, patchsize)

                        output_shard_index = test_count //test_examples_shard
                        test_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                        
                        test_count+=1
                    elif(train_count<num_shards*train_examples_shard):
                        imageName="img_"+str(train_count)+".png"
                        tf_example = create_tf_example(imageName, encodedImg, targets, patchsize)

                        output_shard_index = train_count //train_examples_shard
                        train_tfrecords[output_shard_index].write(tf_example.SerializeToString())

                        train_count+=1
                    else:
                        print("Done")