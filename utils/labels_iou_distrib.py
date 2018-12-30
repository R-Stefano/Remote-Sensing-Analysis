import numpy as np 
import cv2
import matplotlib.pyplot as plt

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

img=cv2.imread("12TVL160640-CROP_Annotated_Cars.png")

box_size=30
#Create labels
labels=np.max(img, axis=-1)

#shape num_targets,2
targets=np.argwhere(labels!=0).astype(float)

#The format that I need is ymin, xmin, ymax, xmax normalized and the class starting from 0 and class name
targets=np.concatenate((targets-box_size/2, targets+box_size/2), axis=-1)

#Retrieve higher IOU (excepted itself) for each target
results=[]
for idx,target in enumerate(targets):
    ious_me=iou(target, targets)
    results.append(np.sort(ious_me)[-2])

#Retrieve boxes that have a IOU>0.3 with some other box
boxes_idxs=np.argwhere(np.array(results)>0.45)
display_boxes=targets[boxes_idxs]

showImg=cv2.imread("12TVL160640-CROP.png")

for b in np.squeeze(display_boxes):
    cv2.rectangle(showImg, (int(b[1]), int(b[0])), (int(b[3]), int(b[2])), (0,0,255))
cv2.imwrite("testing.png",showImg)

n, bins, patches =plt.hist(results, bins=20, rwidth=0.95)
plt.show()

print(n)
