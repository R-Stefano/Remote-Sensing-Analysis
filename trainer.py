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
    batch_size=5
    myfile=tf.summary.FileWriter("tensorboard/", sess.graph)

    for _,_,inputList in os.walk("trainData"):
        data=sorted(inputList)

    try:
        global_step=0
        #REMEMBER TO LOAD BACK THE GLOBAL STEP
        while (True):
            #load input images
            for idx,file in enumerate(data):
                mydict=pickle.load(open("trainData/"+file, "rb"))
                inputData=mydict["X"]
                labelData=mydict["Y"]

                #Train set
                if (idx!=len(data)-1):
                    for startB in range(0,inputData.shape[0], batch_size):
                        endB=startB+batch_size

                        xBatch=inputData[startB:endB]
                        yBatch=creatingLabels(labelData[startB:endB], 416,[13,26], batch_size)

                        _,summ,deb1,deb2,deb3, deb4=sess.run([model.opt, model.trainData,    model.output, model.box_loss, model.x_boxes, model.y_coords], feed_dict={model.input:xBatch/255., model.Y: yBatch})
                        myfile.add_summary(summ, global_step=global_step)
                        #print("predictions - min: {:.2f}, max: {:.2f}".format(np.min(deb1[:,:,2:4]),np.max(deb1[:,:,2:4])))
                        #print("labels      - min: {:.2f}, max: {:.2f}".format(np.min(yBatch[:,:,2:4]),np.max(yBatch[:,:,2:4])))
                        #print("pred boxes  - min: {:.2f}, max: {:.2f}".format(np.min(deb3), np.max(deb3)))
                        #print("loss boxes  - min: {:.2f}, max: {:.2f}".format(np.min(deb2), np.max(deb2)))
                        #print("loss box shape", deb2.shape)
                        #print("\n\n")
                        global_step+=1
                #use the last batch as testing
                else:
                    for startB in range(0,inputData.shape[0], batch_size):
                        endB=startB+batch_size

                        xBatch=inputData[startB:endB]
                        yBatch=creatingLabels(labelData[startB:endB], 416,[13,26], batch_size)

                        _,summ=sess.run([model.loss, model.testData], feed_dict={model.input:xBatch/255., model.Y: yBatch})
                        myfile.add_summary(summ, global_step=global_step)
                        global_step+=1
    except (KeyboardInterrupt,SystemExit):
        model.saver.save(sess,"myModel/yolo-tiny_graph.ckpt")

