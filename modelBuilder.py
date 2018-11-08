from modelUtils import convolution, upsample
import tensorflow as tf

def tiny_yolov3(input):
    for i in range(6):
        x=convolution(input, 16*(pow(2,i)), [3,3])

        if i==4:
            route_1=x
        
        if i==5:
            x=tf.layers.max_pooling2d(x, [2,2], strides=1, padding="SAME")
        
        else:
            x=tf.layers.max_pooling2d(x, [2,2], strides=1)
    
    x=convolution(x, 1024, [3,3])
    x=convolution(x, 256, [1,1])
    route_2=x

    #x=convolution(x, 512, [3,3])

    #Detection 1
    #pred1=detection_layer(x)

    x=convolution(route_2, 128, [1,1])

    newShape=route_1.get_shape().as_list()
    x=upsample(x, newShape)
    #concat along channels
    x=tf.concat((x, route_1), axis=3)

    x=convolution(x, 256, [3,3])

    #Detection 2
    #pred2=detection_layer(x)

    return x#pred1, pred2




