import numpy as np

def iou(box1, box2):   
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

def filterPreds(predictions,imgsize, threshold, iou_threshold):
    '''
    predictions:
        Is a matrix of shape [N, preds, 5+classes].
        the preds dimension has the predictions for each box at each scale
    imgsize:
        A number that indicates the H and W of the original image
    threshold:
        threshold is the number which greater confidences value are retrieved
    iou_threshold:
        number which lower iou values are kept and greater are discarded
    '''

    #Return True where conf score higher then threshold, otherwise false
    mask=predictions[:,:,4]>threshold
    #Set the predictions where in the mask is False to 0
    masked_preds=predictions*mask[:,:,None]

    #Convert x,y, w, h to  vertices: x_0,y_0 and x_1,y_1
    box_vertixes=np.zeros((masked_preds[:,:,:4].shape))
    w,h=masked_preds[:,:,2]/2, masked_preds[:,:,3]/2
    box_vertixes[:,:,0]=(masked_preds[:,:,0] - w)
    box_vertixes[:,:,1]=(masked_preds[:,:,1] - h)
    box_vertixes[:,:,2]=(masked_preds[:,:,0] + w)
    box_vertixes[:,:,3]=(masked_preds[:,:,1] + h)

    masked_preds[:,:,:4]=box_vertixes

    results=[]
    #Iterate among the batch size. An image at the time
    for i, img in enumerate(masked_preds):
        #Obtain the indexes where the preds are not 0
        non_zero_idxs=np.nonzero(img)

        #Retrieve the predictions that are not 0
        preds=img[non_zero_idxs].reshape(-1, img.shape[-1])

        #For every prediction, retrieve the idx of the class with higher value
        classes_preds=np.argmax(preds[:,5:], axis=1)

        #Retrieve the unique classes from the predictions
        unique_classes = list(set(classes_preds.reshape(-1)))

        img_results={}
        #Perform Non-max suprression for each class
        for clas in unique_classes:
            #Retrieve the predictions of the clas
            clas_mask = classes_preds == clas
            clas_preds = preds[np.nonzero(clas_mask)]
            
            #Sort the predictions based on the confidence
            #.argsort() puts on top the lowest value, so we flip the array using [::-1]
            conf_sorted_idxs=clas_preds[:,4].argsort()[::-1]
            sorted_preds=clas_preds[conf_sorted_idxs]

            #Retrieve the bounding boxes attrs
            clas_boxes=sorted_preds[:,:4]
            #Retrieve the conf score attr
            clas_conf=sorted_preds[:,4]

            #Compute the IOU for a pred with all the other predictions
            while len(clas_boxes)>0:
                #Retrieve the pred with higher confidence of the class
                box=clas_boxes[0]
                conf=clas_conf[0]
                
                #If a new class, save the prediction
                if clas not in img_results:
                    img_results[clas]=[]

                img_results[clas].append([box, conf])

                #Retrieve all the following predictions
                clas_boxes=clas_boxes[1:]
                clas_conf=clas_conf[1:]

                #Compute the Intersection of Union of 2 boxes. 
                #What it does is to compute the area where the 2 boxes overlap
                #and then divide it by the sub of the areas minux the intersection area
                #iou=intArea(A1+A2-intArea)
                ious=np.asarray([iou(box,b) for b in clas_boxes])
                
                #if the iou is under the threshold keep it, otherwise set to 0.
                #Get rid of the predictions of the same class in the same place (High IOU)
                #and keep only the ones of the same class but in different places in the image
                iou_mask=ious<iou_threshold
                clas_boxes=clas_boxes[np.nonzero(iou_mask)]
                clas_conf=clas_conf[np.nonzero(iou_mask)]
        
        results.append(img_results)
    
    return results
