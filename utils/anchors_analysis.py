import numpy as np
import cv2
base_box=[0,0,38,38]
anchors=[8,16,32,64,128]

ratios=[(2,1),(1,1),(1,2)]

anchors_scales=[i*0.1 for i in range(1,100)]
octaves=[i for i in range(1,7)]

results={}

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

for anc in anchors_scales:
    for oc in octaves:
        #generate epoch anchorss
        anc_copy=anchors.copy()
        #scale
        for idx, a in enumerate(anc_copy):
            anc_copy[idx]*=anc
        anc_octaves=[]
        #create octaves
        for ancCopy in anc_copy:
            for i in range(oc):
                anc_octaves.append(ancCopy*(2**i/oc))
        final=[]
        for r in ratios:
            for j in anc_octaves:
                box=[0,0,r[0]*j, r[1]*j]
                final.append(box)

        #compute IOUS with base_boxù
        ious=iou(np.array(base_box), np.array(final))
        #how many boxes have a IOU>0.45
        avg_ious=len(ious[ious>0.45])

        if avg_ious in results:
            results[avg_ious]={"anchors_scale":anc,
                               "octaves":oc,
                               "prev":results[avg_ious]}
        else:
            results[avg_ious]={"anchors_scale":anc,
                               "octaves":oc}

img=np.ones((1000,1000,3))*255

#depict base box
cv2.rectangle(img, (int(base_box[1]), int(base_box[0])), (int(base_box[3]), int(base_box[2])), (0,0,255))

#sort the configurations based on IOU
idxs=list(results.keys())
idxs.sort(reverse=True)

#print top 3 configurations
for i in range(1,6):
    print("Configuration: {} num boxes: {:.2f}".format(i,idxs[i]))
    print(results[idxs[i]])
    print("\n")

settings=results[idxs[0]]
for a in anchors:
    for rat in ratios:
        for oc in range(int(settings["octaves"])):
            h=a*rat[0]*settings["anchors_scale"]*(2**(oc/int(settings["octaves"])))
            w=a*rat[1]*settings["anchors_scale"]*(2**(oc/int(settings["octaves"])))
            box=[0,0,h,w]
            cv2.rectangle(img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255,0,0))

cv2.imwrite("boxes.png", img)


            
