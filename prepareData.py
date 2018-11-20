import numpy as np
import pickle
import os
import cv2

patchsize=500
stride=400
imgsize=416
num_classes=1
fileDim=500 #260mb per file

img_count=0

inputData=[]
labelsList=[]

#every image has composed by 3 files: image, positive, negative
for _,_,fileList in os.walk("images"):
    sortedFiles=sorted(fileList)
    for i in range(0,len(sortedFiles), 3):
        print("check image name:",sortedFiles[i])
        image=cv2.imread("images/"+sortedFiles[i])
        posAnn=cv2.imread("images/"+sortedFiles[i+1])

        #Slide vertically
        for sH in range(0, image.shape[0], stride):
            eH=sH+patchsize
            #Slide horizontally
            for sW in range(0,image.shape[1], stride):
                eW=sW+patchsize

                #Collect the patch from the image
                patch=image[sH:eH,sW:eW]
                patchPosAnnotated=posAnn[sH:eH,sW:eW]

                patch=cv2.resize(patch, (imgsize,imgsize))
                patchPosAnnotated=cv2.resize(patchPosAnnotated, (imgsize,imgsize))

                #Create labels
                labels=np.max(patchPosAnnotated, axis=-1)

                targets=np.argwhere(labels!=0)

                #create box size labels
                box_size=np.tile([50,50], (targets.shape[0],1))

                #create isobject label
                isObj=np.ones((targets.shape[0],1))

                #isobj, x, y , w, h
                y_true=np.concatenate((isObj,targets,box_size), axis=-1)

                if(img_count%fileDim==0 and img_count!=0):
                    mydict={"X":np.asarray(inputData), "Y":labelsList}
                    with open("trainData/batch_"+str(img_count//fileDim)+".pickle", "wb") as f:
                        pickle.dump(mydict,f) 

                    inputData=[]
                    labelsList=[]

                #Save input image
                inputData.append(patch)
                #save labels
                labelsList.append(y_true)
                img_count+=1