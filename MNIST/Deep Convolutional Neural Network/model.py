import tensorflow as tf

def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def pool2D(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def init_weights(shape):
    rand = tf.truncated_normal(shape, stddev=0.1)
    initial_weight = tf.Variable(rand)
    return initial_weight

def init_biases(shape):
    const = tf.constant(0.1, shape=shape)
    initial_bias = tf.Variable(const)
    return initial_bias

#Based on an input 'x', this function tells us what the network should output as output 'o'
def model(x, keep_prob):
    x = tf.reshape(x, [-1, 28, 28, 1])
    w_conv1 = init_weights([5, 5, 1, 32])
    b_conv1 = init_biases([32])
    
    w_conv2 = init_weights([5, 5, 32, 64])
    b_conv2 = init_biases([64])
    
    w_fully_connected = init_weights([7 * 7 * 64, 1024])
    b_fully_connected = init_biases([1024])
    
    w_output = init_weights([1024, 10])
    b_output = init_biases([10])

    # CONVOLUTION + POOLING
    conv1_layer = tf.nn.relu(conv2D(x, w_conv1) + b_conv1)
    pool1_layer = pool2D(conv1_layer)

    # CONVOLUTION + POOLING
    conv2_layer = tf.nn.relu(conv2D(pool1_layer, w_conv2) + b_conv2)
    pool2_layer = pool2D(conv2_layer)

    # FULLY CONNECTED
    pool2_layer = tf.reshape(pool2_layer, [-1, 7*7*64])
    fully_connected_layer = tf.nn.relu(tf.matmul(pool2_layer, w_fully_connected) + b_fully_connected)
    
    # DROPOUT
    tf.nn.dropout(fully_connected_layer, keep_prob)
    
    # OUTPUT
    output = tf.matmul(fully_connected_layer, w_output) + b_output
    return output