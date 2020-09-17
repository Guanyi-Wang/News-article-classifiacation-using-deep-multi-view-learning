import tensorflow as tf
import image_data_handler
from text_data_handler import batch_iterator
import time
import os
from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss, coverage_error, hamming_loss
from print_all_variables import print_all_variables

# Define parameters
num_training = 73800
feature_length = 4096
num_labels = 8

# Get data
x, y = image_data_handler.load_image_data()
# Separate training and testing set
x_train = x[:num_training]
x_test = x[num_training:]
y_train = y[:num_training]
y_test = y[num_training:]

print(x_train.shape)
print(y_train.shape)


class ImageNN(object):
    def __init__(self, num_features, num_labels, dropout_keep_prob, num_hidden_neuron):
        # Define placeholder
        self.input_x = tf.placeholder(tf.float32, [None, num_features], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_labels], name='input_y')
        # # Add dropout layer
        # with tf.name_scope("dropout"):
        #     self.dropout = tf.nn.dropout(self.input_x, dropout_keep_prob)

        with tf.name_scope("hidden_layer"):
            W_hidden = tf.Variable(tf.truncated_normal([num_features, num_hidden_neuron], stddev=0.1), name='W_hidden')
            b_hidden = tf.Variable(tf.truncated_normal([num_hidden_neuron]), name="b_hidden")
            self.regularizer_hidden = tf.nn.l2_loss(W_hidden, name="regularizer_hidden")
            self.output_hidden = tf.nn.relu(tf.nn.xw_plus_b(self.input_x, W_hidden, b_hidden, name='output_hidden'))
        # Define weights and bias
        with tf.name_scope("output"):
            W_output = tf.Variable(tf.truncated_normal([num_hidden_neuron, num_labels], stddev=0.1), name='W_output')
            b_output = tf.Variable(tf.truncated_normal([num_labels]), name="b_output")
            self.regularizer_output = tf.nn.l2_loss(W_output, name="regularizer_output")
            self.output = tf.nn.xw_plus_b(self.output_hidden, W_output, b_output, name='output')
            self.scores = tf.nn.sigmoid(self.output, name='scores')
        # Define loss function
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.input_y, name='losses')
            self.loss = tf.reduce_mean(losses+0.01*self.regularizer_hidden+0.01*self.regularizer_output)

        # # Calculate average precision
        # with tf.name_scope('average_precision'):
        #     try:
        #         self.average_precision = label_ranking_average_precision_score(self.input_y, self.scores)
        #     except:
        #         print(self.input_y)

########################################
# Training

# Define training parameters
epochs = 50
batch_size = 256

# Create session
with tf.Graph().as_default():
    with tf.Session() as sess:
        image_nn = ImageNN(num_labels=num_labels,
                           num_features=feature_length,
                           num_hidden_neuron=30,
                           dropout_keep_prob=0.5)

        # Define training operation for gradient update
        global_step = tf.Variable(0, name="global_step", trainable=False)  # a counter to count on steps
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(image_nn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Generate output directory
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "image_nn_runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Generate summaries
        loss_summary = tf.summary.scalar("loss", image_nn.loss)
        #ap_summary = tf.summary.scalar("average_precision", ImageNN.average_precision)
        # rank_summary = tf.scalar_summary("ranking_loss", text_cnn.ranking_loss)
        # coverage_summary = tf.scalar_summary("coverage_error", text_cnn.coverage)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test summaries
        test_summary_op = tf.summary.merge([loss_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Generate checkpoint
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        # Tensorflow assumes this directory already exists so we need to create it
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                image_nn.input_x: x_batch,
                image_nn.input_y: y_batch,

            }

            #print("shape of y_batch:{}".format(y_batch.shape()))
            _, step, summaries, loss, scores = sess.run(
                [train_op, global_step, train_summary_op, image_nn.loss, image_nn.scores],
                feed_dict)
            train_summary_writer.add_summary(summaries, step)


        def test_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a test set
            """
            feed_dict = {
                image_nn.input_x: x_batch,
                image_nn.input_y: y_batch,
            }
            step, summaries, loss, scores = sess.run(
                [global_step, test_summary_op, image_nn.loss, image_nn.scores],
                feed_dict)
            average_precision = label_ranking_average_precision_score(y_test, scores)
            ranking_loss = label_ranking_loss(y_test, scores)
            coverage = coverage_error(y_test, scores)
            prediction = scores
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            hammingloss = hamming_loss(y_test, prediction)
            print("{}: step {}, loss {:g}, AP {:g}, ranking_loss {:g}, coverage {:g}, hamming_loss {:g} ".format('dev',
                                                                                                                 step,
                                                                                                                 loss,
                                                                                                                 average_precision,
                                                                                                                 ranking_loss,
                                                                                                                 coverage,
                                                                                                                 hammingloss))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batch
        batches = batch_iterator(list(zip(x_train, y_train)), batch_size, epochs)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)

            current_step = tf.train.global_step(sess, global_step)
            if current_step % 500 == 0:
                print("\nEvaluation:")
                test_step(x_test, y_test, writer=test_summary_writer)
                print()
            if current_step % 500 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)  # path will be returned
                print("Saved model checkpoint to {}\n".format(path))

        print_all_variables(train_only=False)