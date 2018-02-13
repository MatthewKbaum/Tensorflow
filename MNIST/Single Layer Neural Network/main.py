import tensorflow as tf
import model
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/temp/data", one_hot=True)
print('MNIST loaded')

epoch_num = 10
batch_size = 100

# This will create the complete graph, that allows us
# to both train and test our code
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

predict_y = model.model(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict_y,labels=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

anecdote_test = tf.argmax(predict_y, 1)

correct = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver()

# train_network() will train a network to solve
def train_network():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            for _ in range(int(mnist.train.num_examples/batch_size)):
                train_x,train_y = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: train_x, y: train_y})
            print("***")
            print('Epoch: ',epoch+1)
            print('Accuracy: ',sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels})*100,'%')
            print("***")
            saver.save(sess, "./NETWORK_MODEL/model.ckpt")

def test_network():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./NETWORK_MODEL/model.ckpt")
        print('Testing Accuracy:',sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})*100,'%')

def test_anecdote(x):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./NETWORK_MODEL/model.ckpt")
        output = sess.run(anecdote_test, feed_dict={x: x})
    return output
        
#Training the graph
train_network()

#Testing the graph
test_network()