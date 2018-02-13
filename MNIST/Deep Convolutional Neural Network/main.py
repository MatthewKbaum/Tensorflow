import tensorflow as tf
import model
from tensorflow.examples.tutorials.mnist import input_data

checkpoint_path = "./NETWORK_MODEL/model"

mnist = input_data.read_data_sets("/temp/data", one_hot=True)
print('MNIST loaded')

total_epochs = 10
batch_size = 50

tf.reset_default_graph()

# This will create the complete graph, that allows us
# to both train and test our code
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

predict_y = model.model(x, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict_y,labels=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

anecdote_test = tf.argmax(predict_y, 1)

correct = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver()

# train_network() will train a network to solve the loss function
def train_network():
    print("Training begun")
    with tf.Session() as sess:
        print("Session started")
        sess.run(tf.global_variables_initializer())
        print("Parameters initialized.")
        for i in range(total_epochs):
            for j in range(int(mnist.train.num_examples / batch_size)):
                batch = mnist.train.next_batch(batch_size)
                train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
            saver.save(sess, checkpoint_path)
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            print("***")
            print('Epoch: ', i+1)
            print('Accuracy: ',sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
            print("***")

def test_network():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)
        print('Testing Accuracy:',sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})*100,'%')

def test_anecdote(x):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)
        output = sess.run(anecdote_test, feed_dict={x: x})
    return output

#Training the graph
train_network()

#Testing the graph
#test_network()