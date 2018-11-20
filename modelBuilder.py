from modelUtils import convolution, upsample, predictionLayer
import tensorflow as tf
import numpy as np
import os

class Yolov3_tiny:
    def __init__(self, num_classes, imgsize, anchors):
        self.input=tf.placeholder(tf.float32, [None, imgsize, imgsize, 3], name="RGB_image")
        self.num_classes=num_classes
        self.anchors=anchors
        self.num_anchors=3    
        self.imgDim=imgsize

        self.buildModel()
        self.buildTraining()
        self.buildSummary()
        self.saver=tf.train.Saver()


    def buildModel(self):
        x=self.input
        for i in range(6):
            x=convolution(x, 16*pow(2,i), [3,3])

            if i==0:
                self.endConv1=x

            if i==4:
                route_1=x
            
            if i==5:
                x=tf.layers.max_pooling2d(x, [2,2], strides=1, padding="SAME")
            
            else:
                x=tf.layers.max_pooling2d(x, [2,2], strides=2)
        
        x=convolution(x, 1024, [3,3])
        x=convolution(x, 256, [1,1])
        route_2=x

        x=convolution(x, 512, [3,3])

        with tf.variable_scope("prediction_1"):
            self.pred1, self.deb1=predictionLayer(x,self.num_anchors, self.num_classes,self.anchors[1],self.imgDim)

        x=convolution(route_2, 128, [1,1])

        newShape=route_1.get_shape().as_list()
        x=upsample(x, newShape)
        #concat along channels
        x=tf.concat((x, route_1), axis=-1, name="concatenate_featureMaps")

        x=convolution(x, 256, [3,3])

        with tf.variable_scope("prediction_2"):
            self.pred2,self.deb2=predictionLayer(x,self.num_anchors, self.num_classes,self.anchors[0],self.imgDim)

        #Concatenate the predictions at different scales
        self.output=tf.concat((self.pred1,self.pred2),axis=1)

    
    def buildTraining(self):
        #Labels has shape N, preds, 5 where 5=[isobj, x,y,w,h]
        self.Y=tf.placeholder(tf.float32, name="labels")

        objects,y_coords,y_boxes=tf.split(self.Y, [1,2,2], axis=-1)
        x_coords, x_boxes, x_conf, x_classes=tf.split(self.output, [2,2,1,self.num_classes], axis=-1)

        #Normalize x,y,w,h by the input image size
        print(x_coords)
        #Compute x,y loss
        xy_loss=tf.reduce_sum(tf.square(y_coords-x_coords), axis=[1,2])*objects
        print("Should be shape N, preds, 1", tf.shape(xy_loss))
        #Compute boxsize loss
        box_loss=tf.reduce_sum(tf.square(tf.sqrt(y_boxes)-tf.sqrt(x_boxes)), axis=[1,2])*objects
        print("Should be shape N, preds, 1", box_loss)
        #Compute confidence
        conf_loss=tf.reduce_sum(tf.log(x_conf), axis=[1,2])*objects + tf.reduce_sum(tf.log(1-x_conf), axis=[1,2])*(1-objects)

        #compute class pred
        class_loss=tf.reduce_sum(tf.log(x_classes+1e-9), axis=[1,2])*objects


        self.loss=tf.reduce_mean(xy_loss+box_loss+conf_loss+class_loss)

        optimizer=tf.train.AdamOptimizer(0.00001)

        vars1=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="prediction_1")
        vars2=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="prediction_2")
        with tf.variable_scope("optimizer"):
            self.grad_vars=optimizer.compute_gradients(self.loss,var_list=[vars1, vars2])

            self.opt=optimizer.apply_gradients(self.grad_vars)

    def buildSummary(self):
        prevk1=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Conv_8/weights")
        prevk2=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Conv_10/weights")

        self.trainData=tf.summary.merge([tf.summary.scalar("loss", self.loss),
                                         tf.summary.histogram("grads_kernel1", self.grad_vars[0][0]),
                                         tf.summary.histogram("grads_kernel2", self.grad_vars[2][0]),
                                         tf.summary.histogram("prevk1", prevk1),
                                         tf.summary.histogram("prevk2", prevk2)])
        self.testData=tf.summary.merge([tf.summary.scalar("test_loss", self.loss)])

    
    def loadWeights(self, sess):
        #check if exist .ckpt, is yes load it. otherwise comnvert .weights
        if os.path.isfile("myModel/yolo-tiny_graph.ckpt"):
            self.saver.restore(sess, "myModel/yolo-tiny_graph.ckpt")
            print("Loaded tensorflow format")
        else:
            #Initialize adam variables
            sess.run(tf.global_variables_initializer())
            #initialize layer weights
            loadedWeights=self.convertWeightsFormat()
            sess.run(loadedWeights)
            self.saver.save(sess, "myModel/yolo-tiny_graph.ckpt")
            print("Weights loaded from .weights file and saved in TF format")


    def convertWeightsFormat(self):
        weights_file=open("model/yolov3-tiny.weights", "rb")
        #Every int32 number is 4 bytes. So, we retrieve the first 3 numbers with a total of 12 bytes
        major, minor, revision= np.ndarray(shape=(3,), dtype='int32', buffer=weights_file.read(12))
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
        print('### Weights Header ### \nmajor: {} \nminor: {} \nsubversion: {} \nimages seen: {}'.format(major, minor, revision, seen))

        scopes=[]
        var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        
        #Retrive the scopes/layers
        for var in var_list:
            #retrieve layer "scope"
            sc=var.name[:var.name.find("/")+1]
            if sc not in scopes:
                scopes.append(sc)

        assign_ops=[]
        #Iterate through each layer assigning the variables
        for idx, scope in enumerate(scopes):
            if scope=="optimizer/":
                break
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

            filters_shape=var_list[0].get_shape().as_list()

            #Batch norm layer
            if len(var_list)>2:
                #Retrieve Beta
                beta=var_list[2]
                
                #Read weights for beta
                wg = np.ndarray(
                    shape=(filters_shape[-1],),
                    dtype='float32',
                    buffer=weights_file.read(filters_shape[-1] * 4))
                    
                
                assign_ops.append(tf.assign(beta, wg, validate_shape=True))
                
                #Read weights for others norm batch vars: gamma, mean, sigma
                wg=np.ndarray(
                    shape=(3, filters_shape[-1]),
                    dtype='float32',
                    buffer=weights_file.read(3*filters_shape[-1] * 4))
                

                assign_ops.append(tf.assign(var_list[1], wg[0], validate_shape=True))#gamma
                assign_ops.append(tf.assign(var_list[3], wg[1], validate_shape=True))#mean
                assign_ops.append(tf.assign(var_list[4], wg[2], validate_shape=True))#variance

                #Finally, load the weights for the filters in format OIHW
                wg=np.ndarray(shape=[filters_shape[3],filters_shape[2],filters_shape[0],filters_shape[1]],
                            dtype='float32',
                            buffer=weights_file.read(np.product(filters_shape) * 4))

                #Convert to format HWIO
                wg=np.transpose(wg, (2,3,1,0))
                assign_ops.append(tf.assign(var_list[0], wg, validate_shape=True))
            #If not batch norm, first the bias
            else:
                #Retrieve the weights such as that there will be 80 classes even if I use 1
                wg = np.ndarray(
                    shape=((80+5)*3,),
                    dtype='float32',
                    buffer=weights_file.read((80+5)*3 * 4))

                #Assign only the ones that fit my network weights
                assign_ops.append(tf.assign(var_list[1], wg[:filters_shape[-1]], validate_shape=True))

                #Same thing for the filter: 80 classes and then retrieve just for 1
                original_shape=[(80+5)*3,filters_shape[2],filters_shape[0],filters_shape[1]]

                #The filters weights in the prediction layers must be dealed differently
                wg=np.ndarray(shape=original_shape,
                            dtype='float32',
                            buffer=weights_file.read(np.product(original_shape) * 4))

                #Convert to format HWIO
                wg=np.transpose(wg, (2,3,1,0))
                assign_ops.append(tf.assign(var_list[0], wg[:,:,:,:filters_shape[-1]], validate_shape=True))

        return assign_ops



