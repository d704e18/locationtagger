import tensorflow as tf
from machine_learning_models.utils import *

class LogisticRegression:

    def __init__(self):
        input = tf.placeholder(tf.int32, [None, sensors])
        target = tf.placeholder(tf.int32, [None, 4])


    def make_config(self):
        config = AttrDict()
        config.input_size = 25 # todo change to actual input size
        config.learning_rate = 0.001
        config.training_epochs = 50

        return config

    def log_reg_network(self, inputs, config):

        W = tf.Variable(tf.zeros([config.input_size, 4]))
        b = tf.Variable(tf.zeros([4]))

        prediction = tf.nn.softmax(tf.matmul(inputs, W) + b)
        return prediction

    def define_graph(self, config, inputs, targets):
        tf.reset_default_graph()

        prediction = self.log_reg_network(config, inputs)
        # Cost function - Cross entropy / negative log likelihood
        cost = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(prediction), axis=1))
        # Optimization - Adaptive momentum estimation  Adam
        optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)


if __name__ == "__main__":

    learning_rate = 0.01
    sensors = 25
    training_epochs = 100


    prediction = tf.nn.softmax(tf.matmul(input, W) + b)
    # Cost function - Cross entropy / negative log likelihood
    cost = tf.reduce_mean(-tf.reduce_sum(target*tf.log(prediction), axis=1))
    # Optimization - Adaptive momentum estimation  Adam
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
