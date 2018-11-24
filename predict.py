import os
import cv2
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import numpy as np

version="new"
#Directory that containes the exported model
myModel="model_"+version
ckptPath=myModel+"/frozen_inference_graph.pb"

labels="data/labels.pbtxt"

num_classes=1

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
    print("\n****\n")
    print(box)
    print(other_boxes[0])
    #Compute intersection area
    #Use None to keep the dimension and allows concatenation across values
    int_x0=np.max(np.concatenate((box[:,0,None], other_boxes[:,0,None]), axis=-1), axis=-1)
    int_y0=np.max(np.concatenate((box[:,1,None], other_boxes[:,1,None]), axis=-1), axis=-1)
    int_x1=np.min(np.concatenate((box[:,2,None], other_boxes[:,2,None]), axis=-1), axis=-1)
    int_y1=np.min(np.concatenate((box[:,3,None], other_boxes[:,3,None]), axis=-1), axis=-1)
    
    print(int_x0[0])
    print(int_y0[0])
    print(int_x1[0])
    print(int_y1[0])

    int_area=(int_x1-int_x0)*(int_y1-int_y0)

    print(int_area[0])
    print("\n****\n")


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

count=0
#Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        #Grab the nodes on the graph by name
        image_tensor=detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores=detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes=detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections=detection_graph.get_tensor_by_name('num_detections:0')

        count=0
        #open the image and process it
        img=cv2.imread("top_potsdam_3_13_RGB.png")
        imgHeight=img.shape[0]
        imgWidth=img.shape[1]

        boxList=[]
        scoreList=[]
        originalSizeH=0
        for startP in range(0, imgHeight, 400):
            endP=startP+500
            originalSizeW=0
            for startPH in range(0, imgWidth, 400):
                endPH=startPH+500

                patch=img[startP:endP,startPH: endPH]
                patchShape=patch.shape


                inputImg=cv2.resize(patch, (416,416))

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

                '''

                #Traslate the boxes in order to match the output image
                boxes=np.squeeze(boxes)

                #Convert to 500 dimension
                #boxes=boxes*(500/416)
                print("prev max:",np.max(boxes[:,0::2]), " prev min:",np.min(boxes[:,0::2]))
                #translate Vertical
                boxes[:,0::2]=(boxes[:,0::2]*patchShape[0])+originalSizeH
                #Translate Horizontally
                boxes[:,1::2]=(boxes[:,1::2]*patchShape[1])+originalSizeW
                print("max:",np.max(boxes[:,0::2]), "min:",np.min(boxes[:,0::2]))

                #Store the results
                boxList.extend(boxes.tolist())
                scoreList.extend(np.squeeze(scores).tolist())

                '''
                for b in boxes[0]:
                    b=b*416
                    #x=(b[2]-b[0])//2
                    #y=(b[3]-b[1])//2
                    #print(x,",",y)
                    #cv2.circle(inputImg, (int(x),int(y)), 10, (0, 255, 0))
                    cv2.rectangle(inputImg, (int(b[1]), int(b[0])), (int(b[3]), int(b[2])), (0,255,0))
                
                cv2.imwrite("res_"+str(count)+".png", inputImg)
                count+=1
                #img[startP: startP+ patchShape[0],startPH: startPH+patchShape[1]]=cv2.resize(inputImg, (patchShape[1],patchShape[0]))
                '''
                
                #Translate by the dimension of the stride
                originalSizeW+=400

            originalSizeH+=400

        #Apply NMS
        
        #convert back to numpy to work with idxs
        #Normalize boxes in order to iou max 1
        boxes=np.asarray(boxList)
        res=non_max_suppression_fast(boxes, 0.55)
        '''
        #boxes[:,0::2]=boxes[:,0::2]/imgHeight
        #boxes[:,1::2]=boxes[:,1::2]/imgWidth
        scores=np.asarray(scoreList)

        #Sort the predictions based n confidence
        conf_sorted_idx=scores.argsort()[::-1]
        sorted_boxes=boxes[conf_sorted_idx]

        finalBoxes=[]
        while(len(sorted_boxes)>0):
            #Retrieve higher confidence box
            box=sorted_boxes[0]
            finalBoxes.append(box)

            #Retrieve all the following boxes
            other_boxes=sorted_boxes[1:]

            print("aout iou components:")
            print(box)
            print(sorted_boxes[0])
            #Compute IOU
            iouRes=iou(box, other_boxes)

            print("auto iou:", iouRes[0])

            #Get rid of the boxes with high IOU (means same box)
            iou_mask=iouRes<0.6
            print("number non zero:", len(np.nonzero(iou_mask)[0]))
            print("number non zero:", np.nonzero(iou_mask))

            sorted_boxes=sorted_boxes[np.nonzero(iou_mask)]
        '''
        for b in res:
            print(b)
            cv2.rectangle(img, (int(b[1]), int(b[0])), (int(b[3]), int(b[2])), (0,255,0))

        cv2.imwrite("resultNMS_"+version+".png", img)
