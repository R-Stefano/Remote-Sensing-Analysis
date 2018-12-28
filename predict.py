from __future__ import division

import os
import cv2
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import numpy as np

#Directory that containes the exported model
ckptPath="model/model.pb"

input_image_name="top_potsdam_6_8_RGB"

labels="model/labels.pbtxt"

num_classes=1
score_threshold=0.3
patch_size=500
stride=400

#Load model into memory
detection_graph=tf.Graph()
with detection_graph.as_default():
    od_graph_def=tf.GraphDef()
    with tf.gfile.GFile(ckptPath, 'rb') as fid:
        serialized_graph=fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#Creating labels
label_map=label_map_util.load_labelmap(labels)
categories=label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=False)
category_index=label_map_util.create_category_index(categories)


def iou(box, other_boxes):
    #Repeat the box for every example
    box=np.repeat(np.expand_dims(box, axis=0), other_boxes.shape[0] ,axis=0)

    #Compute intersection area
    #Use None to keep the dimension and allows concatenation across values
    int_x0=np.max(np.concatenate((box[:,0,None], other_boxes[:,0,None]), axis=-1), axis=-1)
    int_y0=np.max(np.concatenate((box[:,1,None], other_boxes[:,1,None]), axis=-1), axis=-1)
    int_x1=np.min(np.concatenate((box[:,2,None], other_boxes[:,2,None]), axis=-1), axis=-1)
    int_y1=np.min(np.concatenate((box[:,3,None], other_boxes[:,3,None]), axis=-1), axis=-1)

    int_area=(int_x1-int_x0)*(int_y1-int_y0)

    b1_area=(box[:,2]-box[:,0])*(box[:,3]-box[:,1])
    b2_area=(other_boxes[:,2]-other_boxes[:,0])*(other_boxes[:,3]-other_boxes[:,1])

    iou=int_area/(b1_area + b2_area-int_area+ 1e-09)

    return iou

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

#Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        #Grab the nodes on the graph by name
        image_tensor=detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores=detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes=detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections=detection_graph.get_tensor_by_name('num_detections:0')

        #open the image and process it
        img=cv2.imread(input_image_name+".png")

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

                '''
                I think that the most efficient solution is to store all the predicitons converted into the 
                dimension of the final image. Then, perform NMS on the final list.

                Now, How I convert the coordinates of the patch (416,416) into the (2000,2000) map coords?

                I believe that just traslating the points 
                The scores are already sorted
                
                vis_util.visualize_boxes_and_labels_on_image_array(
                    inputImg,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=1)
                
                cv2.imwrite(input_image_path+"_output_n"+str(count)+".png", inputImg)

                '''
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

        #Apply NMS
        boxes=np.asarray(boxList)
        res=non_max_suppression_fast(boxes, 0.5)

        #Retrieve the ground truth bounding boxes
        labels=cv2.imread(input_image_name+"_Annotated_Cars.png")
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
            ious_mask=ious<0.5


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

        cv2.imwrite(input_image_name+"_output.png", img)
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        F1=2*(precision*recall)/(precision+recall)
        accuracy=(tp)/(tp+fn+fp)
        
        print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, Accuracy: {:.2f}".format(precision,recall,F1, accuracy))
    
