import cv2 
import numpy as np
import os

def showTarget(image, labels):
    #Reduce to 1 channel only
    coords=np.max(labels, axis=-1)

    #Retrieve the coordinates of the targets
    #convert [pos, label] -> [label,pos] (transposing)
    points=np.transpose(np.asarray(np.where(coords==255)))

    #High and width
    shift=np.array([25,25])

    for idx, coord in enumerate(points):
        shiftV1=coord-shift
        shiftV2=coord+shift

        cv2.rectangle(image, (shiftV1[1],shiftV1[0]), (shiftV2[1],shiftV2[0]), (0,0,255))
    
    #cv2.imwrite('test_image.png',image)

    return image


def creatingLabels(y_true, imgsize, scales, batch_size):
    outsList=[]
    for scale in scales:
        #create a tensor of zeros of shape
        y_trues=np.zeros((batch_size, scale, scale, 3, 5))
        stride=imgsize/scale
        for idx, b in enumerate(y_true):
            #Compute which grid cell the label belongs {0,scale}
            gridcell_idxs=(b[:,1:3]//stride).astype(int)
            #retrieve the gridcells where the label belongs and assign the label (no IOU, all boxes same prediction)
            y_trues[idx,gridcell_idxs[:,0],gridcell_idxs[:,1]]=np.repeat(np.expand_dims(b, axis=1), 3, axis=1)
        reshapedY=np.reshape(y_trues, (batch_size, scale*scale*3, 5))
        outsList.append(reshapedY)

    output=np.concatenate((outsList[0],outsList[1]), axis=1)
    return output


def prepareInputImage(image, posAnn, negAnn, imgsize, step, num_classes):
    imagePatches=[]
    imageLabels=[]
    count=0
    #Slide vertically
    for sH in range(0, image.shape[0], step):
        eH=sH+imgsize
        #Slide horizontally
        for sW in range(0,image.shape[1], step):
            eW=sW+imgsize

            #Collect the patch from the image
            patch=image[sH:eH,sW:eW]
            patchPosAnnotated=posAnn[sH:eH,sW:eW]
            patchNegAnnotated=negAnn[sH:eH,sW:eW]

            #If the patch is smaller than the normal image, resize it
            if(patch.shape[0]!=imgsize or patch.shape[1]!=imgsize):
                patch=cv2.resize(patch, (imgsize,imgsize))
                patchPosAnnotated=cv2.resize(patchPosAnnotated, (imgsize,imgsize))
                patchNegAnnotated=cv2.resize(patchNegAnnotated, (imgsize,imgsize))

            #Create labels
            carBoundBox=createLabel(patchPosAnnotated, imgsize, isTarget=True)
            negBoundBox=createLabel(patchNegAnnotated, imgsize, isTarget=False)

            imageLabel=np.concatenate((carBoundBox, negBoundBox), axis=0)

            imagePatches.append(patch)
            imageLabels.append(imageLabel)

    return imagePatches, imageLabels

def prepareData(imgsize, step, num_classes):
    count=0
    inputData=0
    labelData=0
    #every image has composed by 3 files: image, positive, negative
    for _,_,fileList in os.walk("images"):
        for file in sorted(fileList):
            if count==0:
                imagePath=cv2.imread("images/"+file)
                count+=1
            elif count==1:
                posAnnotatedPath=cv2.imread("images/"+file)                
                count+=1
            else:
                negAnnotatedPath=cv2.imread("images/"+file)
                inputData, labelData=prepareInputImage(imagePath, posAnnotatedPath, negAnnotatedPath, imgsize, step, num_classes)
                break
                count=0
        break
        

    return inputData, labelData   