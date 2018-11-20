import tensorflow as tf

layer=tf.contrib.layers

batch_norm_params = {
        'decay': 0.9,
        'epsilon': 1e-05,
        'scale': True,
        'is_training': False}

lky_relu=0.1

def convolution(input, num_maps, filter_size, stride=1):
    x=layer.conv2d(input, num_maps, filter_size, stride, padding="SAME",
                    normalizer_fn=layer.batch_norm,
                    normalizer_params=batch_norm_params,
                    activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=lky_relu))
    return x

def residual_block(input, feat_maps):
    x=convolution(input, feat_maps, [1,1])
    x=convolution(x, feat_maps*2, [3,3])

    x_out=x+input

    return x_out

def upsample(input, newShape):
    shapedInput=tf.image.resize_nearest_neighbor(input, (newShape[1],newShape[2]), name="Upsampling")
    return shapedInput

def predictionLayer(input, num_anchors, num_classes, anchors, imgDim):
    ## Detection kernel ##
    #B(5+C)
    detKernelDim=num_anchors*(5+num_classes)
    #Retrieve dimension of the feature map: N,S,S,..
    inputsize=input.get_shape().as_list()
    #Obtain prediction tensor of shape [N,S,S,B(5+C)]
    predMatrix=layer.conv2d(input, detKernelDim, kernel_size=[1,1],
                            activation_fn=None)
    
    print("\n\nPred matrix", predMatrix)

    #Convert to shape [N, B*S*S, 5+C]
    shapedMatrix=tf.reshape(predMatrix, (-1, inputsize[1]*inputsize[2]*num_anchors, 5+num_classes))

    #Retrieve the different predictions
    box_center, box_dim, confScore, classes=tf.split(shapedMatrix, [2,2,1,num_classes], axis=-1)

    #Used to convert the predictions to the prediction scale. Later, convert back to the scale of input image
    stride=imgDim//inputsize[1]
    ## BOX DIMENSION PREDICTION ##
    with tf.variable_scope("box_dimension"):
        #Convert anchors dimension in order to match the feature map dimension
        anchors=[(a[0]/stride,a[1]/stride) for a in anchors]

        #duplicate the anchors for each element
        anchors=tf.tile(anchors, [inputsize[1]*inputsize[2], 1])

        box_sizes=tf.exp(box_dim) * anchors

        #Convert back to the dimension of the input image
        box_sizes=box_sizes*[stride,stride]


    ## BOX CENTER PREDICTION ##
    with tf.variable_scope("position"):
        #normalize the position
        xyNorm=tf.nn.sigmoid(box_center)

        #creating a SXS grid based on feature map dimension
        x_space=tf.range(inputsize[1], dtype=tf.float32)
        y_space=tf.range(inputsize[2], dtype=tf.float32)
        x_grid, y_grid=tf.meshgrid(x_space, y_space)

        #Format the offset for a single bounding box
        x_y_offset=tf.concat((tf.reshape(x_grid,(-1,1)),tf.reshape(y_grid,(-1,1))), axis=-1)
        #Repeat it to have for ever prediction of each box
        x_y_offset=tf.reshape(tf.tile(x_y_offset, [1,num_anchors]), [1,-1,2])

        #obtain the box center relative to the grid cell. es grid-cell (2,2) has center 2.4,2.8 for example
        box_center=xyNorm+x_y_offset

        #Convert back to the dimension of the input image
        box_center=box_center*[stride,stride]
    
    with tf.variable_scope("confidence"):
        ## CONFIDENCE PREDICTION ##
        normConfidence=tf.nn.sigmoid(confScore, name="confidence_normalized")
    
    with tf.variable_scope("classes"):
        #Use sigmoid instead of softmax because a box could predict more than 1 class
        classes=tf.nn.sigmoid(classes, name="classes_normalized")


    prediction=tf.concat([box_center,box_sizes,normConfidence,classes], -1)

    return prediction, box_center


