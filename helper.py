import cv2 
import numpy as np

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

    cv2.imwrite('test_image.png',image)


def prepareInputImage(image, imgsize):
    step=900
        
    count=0
    #Slide vertically
    for sH in range(0, image.shape[0], step):
        eH=sH+imgsize
        #Slide horizontally
        for sW in range(0,image.shape[1], step):
            eW=sW+imgsize

            patch=image[sH:eH,sW:eW]

            #Resize to 416,416
            cv2.imwrite("trainingImages/image_"+str(count)+".png", patch)
            count+=1

