import tensorflow as tf

#Based on an input 'x', this function tells us what the network should output as output 'o'
def model(x):
    w = tf.Variable(tf.random_normal([784,10]))
    b = tf.Variable(tf.random_normal([10]))
    o = tf.add(tf.matmul(x, w), b)
    return o