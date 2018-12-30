from __future__ import division

import os
import cv2
import tensorflow as tf
import numpy as np
import argparse

#Directory that containes the exported model and labels
ckptPath="model/model.pb"

def iou(box, other_boxes):
    #Repeat the box for every example
    box=np.repeat(np.expand_dims(box, axis=0), other_boxes.shape[0] ,axis=0)

    #Compute intersection area
    #Use None to keep the dimension and allows concatenation across values
    int_x0=np.max(np.concatenate((box[:,0,None], other_boxes[:,0,None]), axis=-1), axis=-1)
    int_y0=np.max(np.concatenate((box[:,1,None], other_boxes[:,1,None]), axis=-1), axis=-1)
    int_x1=np.min(np.concatenate((box[:,2,None], other_boxes[:,2,None]), axis=-1), axis=-1)
    int_y1=np.min(np.concatenate((box[:,3,None], other_boxes[:,3,None]), axis=-1), axis=-1)
    
    int_width=np.max(np.concatenate((np.expand_dims(int_x1 - int_x0 + 1, axis=0), np.zeros((1,other_boxes.shape[0]))), axis=0),axis=0)
    int_height=np.max(np.concatenate((np.expand_dims(int_y1 - int_y0 + 1, axis=0), np.zeros((1,other_boxes.shape[0]))), axis=0),axis=0)

    int_area =int_width*int_height

    b1_area=(box[:,2]-box[:,0] +1)*(box[:,3]-box[:,1] +1)
    b2_area=(other_boxes[:,2]-other_boxes[:,0] +1)*(other_boxes[:,3]-other_boxes[:,1] +1)

    iou=int_area/(b1_area + b2_area-int_area+ 1e-9)

    return iou

def NMS(boxes, threshold):
    results=[]
    while len(boxes)>0:
        box=boxes[0]
        results.append(box)
        ious=iou(box, boxes)
        ious_mask=ious<threshold
        boxes=boxes[ious_mask]
    return np.array(results)

#Detection
def detect(detection_graph, img, score_threshold, iou_threshold):
    patch_size=500
    stride=400

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            #Grab the nodes on the graph by name
            image_tensor=detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores=detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes=detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections=detection_graph.get_tensor_by_name('num_detections:0')

            imgHeight=img.shape[0]
            imgWidth=img.shape[1]

            boxList=[]

            originalSizeH=0
            for startP in range(0, imgHeight, stride):
                endP=startP+500
                originalSizeW=0
                for startPH in range(0, imgWidth, stride):
                    endPH=startPH+500

                    inputImg=img[startP:endP,startPH: endPH]

                    patchSize=inputImg.shape

                    #Feed the image and obtain the results
                    (boxes, scores, classes, num)=sess.run([
                        detection_boxes,detection_scores,detection_classes,num_detections
                    ], feed_dict={image_tensor:np.expand_dims(inputImg, axis=0)})

                    #Retrieve the predictions where the score is higher than the threshold
                    scores=np.squeeze(scores)
                    score_mask=scores>score_threshold
                    score_idxs=np.nonzero(score_mask)

                    boxes=np.squeeze(boxes)[score_idxs]

                    #Boxes contains x0,y0,x1,y1 of the bounding boxes in range 0-1
                    #CONVERT BACK TO THE COORDINATES OF THE PATCH
                    boxes[:,0::2]=boxes[:,0::2]*patchSize[0]+originalSizeH
                    boxes[:,1::2]=boxes[:,1::2]*patchSize[1]+originalSizeW

                    boxList.extend(boxes)

                    #Translate by the dimension of the stride
                    originalSizeW+=stride

                originalSizeH+=stride
            
            return np.asarray(boxList)

def metrics(boxes, img, labels, iou_threshold):
    res=NMS(boxes, iou_threshold)

    #format labels
    labels=np.max(labels, axis=-1)
    coords=np.argwhere(labels!=0)
    #convert center to bounding boxes
    coords=np.concatenate((coords-15, coords+15), axis=-1)

    tp=0
    fn=0
    fp=0

    while (len(coords)>0):
        coord=coords[0]
        ious=iou(coord, res)
        ious_mask=ious<iou_threshold


        #Correctly predicted
        if(len(ious_mask[ious_mask==False])!=0):
            cv2.rectangle(img, (int(coord[1]), int(coord[0])), (int(coord[3]), int(coord[2])), (0,255,0))
            #Remove the predictions
            res=res[ious_mask]
            tp+=1
        #No predicted
        else:
            cv2.rectangle(img, (int(coord[1]), int(coord[0])), (int(coord[3]), int(coord[2])), (255,0,0))
            fn+=1

        coords=np.delete(coords, 0,0)

    #Wrong predictions
    for r in res:
        cv2.rectangle(img, (int(r[1]), int(r[0])), (int(r[3]), int(r[2])), (0,0,255))
        fp+=1

    cv2.imwrite("result.png", cv2.resize(img,(1280,1280)))
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    F1=2*(precision*recall)/(precision+recall)
    accuracy=(tp)/(tp+fn+fp)
    
    print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, Accuracy: {:.2f}".format(precision,recall,F1, accuracy))
    print("Total cars: {}, Correctly predicted: {}, Not predicted: {}, wrongly predicted: {}".format((tp+fn), tp, fn, fp))
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("image", help="pass the input image name without .png")
    parser.add_argument('-s','--score_threshold', help="Set the score threshold (default: 0.4)", default=0.4)
    parser.add_argument("-iou", '--iou_threshold', help="Set the IOU threshold (default 0.4)", default=0.4)

    args=parser.parse_args()

    #Load model into memory
    detection_graph=tf.Graph()
    with detection_graph.as_default():
        od_graph_def=tf.GraphDef()
        with tf.gfile.GFile(ckptPath, 'rb') as fid:
            serialized_graph=fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    #open the image
    img=cv2.imread(args.image+".png")

    #Retrieve the ground truth bounding boxes
    labels=cv2.imread(args.image+"_Annotated_Cars.png")

    #Detect cars 
    detections=detect(detection_graph, img, args.score_threshold, args.iou_threshold)
    metrics(detections, img, labels, args.iou_threshold)