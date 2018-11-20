import os
import pickle
import numpy as np
import tensorflow as tf
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


def train(model, sess):
    batch_size=50
    file=tf.summary.FileWriter("tensorboard/", sess.graph)

    for _,_,inputList in os.walk("trainData"):
        data=sorted(inputList)

    #load input images
    for file in data:
        mydict=pickle.load(open("trainData/"+file, "rb"))
        inputData=mydict["X"]
        labelData=mydict["Y"]


        for startB in range(0,inputData.shape[0], batch_size):
            endB=startB+batch_size

            xBatch=inputData[startB:endB]
            yBatch=creatingLabels(labelData[startB:endB], 416,[13,26], batch_size)

            _,_,err,deb=sess.run([model.opt1, model.opt2, model.loss,model.deb1], feed_dict={model.input:inputData/255., model.Y: yBatch})
            print(err)
            #file.add_summary(summ)
