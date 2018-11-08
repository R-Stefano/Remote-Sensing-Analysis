import tensorflow as tf

layer=tf.contrib.layers

def convolution(input, num_maps, filter_size, stride=1):
    x=layer.conv2d(input, num_maps, filter_size, stride, padding="SAME",
                    normalizer_fn=tf.layers.batch_normalization,
                    activation_fn=tf.nn.leaky_relu)

    return x

def residual_block(input, feat_maps):
    x=convolution(input, feat_maps, [1,1])
    x=convolution(x, feat_maps*2, [3,3])

    x_out=x+input

    return x_out

def upsample(input, newShape):
    shapedInput=tf.image.resize_nearest_neighbor(input, (newShape[1],newShape[2]), name="Upsampling")
    return shapedInput