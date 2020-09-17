import tensorflow as tf
from text_data_handler import *
from text_data_handler import batch_iterator
import time
import os
from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss, coverage_error, hamming_loss
import numpy as np

class Clstm_Text(object):

    def __init__(self, sentence_length, num_labels, embedding_size, filter_size, num_filters, num_units, regular_const,dropout_keep_prob,
                 embedding_matrix=[]):

        # Placeholders for input output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_labels], name='input_y')

        # Embedding Layer
        with tf.device('/cpu:0'), tf.name_scope('embedding_layer'):  # force to run on CPU
            word_embedding = tf.Variable(embedding_matrix, trainable='False',
                                         name='embedding_matrix')  # embedding matrix
            # real embedding operation, returns a 3-dimensional tensor:[None, sentence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(word_embedding, self.input_x)
            # expand 1 dimension(channel) :[batch:None, width:sentence_length, height:embedding_size, channel:1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)   # expand 1 dimension at the last
        # Add dropout layer
        with tf.name_scope("dropout_embedding"):
            self.dropout_embedding = tf.nn.dropout(self.embedded_chars_expanded, dropout_keep_prob)

        # Convolution and Max-Pooling Layers
        with tf.name_scope('conv_layer_%s' % filter_size):
            # Convolution layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            filter_matrix = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                        name='filter_matrix')   # filter matrix
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")   # bias term
            conv = tf.nn.conv2d(self.dropout_embedding, filter_matrix, strides=[1, 1, 1, 1], padding='VALID',
                                name='convolution')  # shape[batch, sequence_length - filter_size + 1, 1, num_filters]
            # Apply nonlinear function ReLu(rectified linear unit)
            self.feature = tf.nn.relu(tf.nn.bias_add(conv, b), name='feature')
            # Flatten dimension
            self.feature = tf.reshape(self.feature, [-1, sentence_length-filter_size+1, num_filters])
        # Define LSTM layer
        with tf.name_scope("lstm"):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, state_is_tuple=True)
            self.lstm_outputs, self.last_state = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32,
                                                                       inputs=self.feature)
            # outputs:[batch_size, num_steps, num_unit], last_state:[batch_size, num_unit] for current and hidden state
        # Add dropout layer
        with tf.name_scope("dropout"):
            self.dropout = tf.nn.dropout(self.last_state[0], dropout_keep_prob)

        # Define weights and bias of fully connected layer
        with tf.name_scope("fully_connected"):
            W_output = tf.Variable(tf.truncated_normal([num_units, num_labels], stddev=0.1), name='W_output')
            b_output = tf.Variable(tf.truncated_normal([num_labels]), name="b_output")
            self.regularizer_output = tf.nn.l2_loss(W_output, name="output_layer_regularizer")
            self.output = tf.nn.xw_plus_b(self.dropout, W_output, b_output, name='output')
            self.scores = tf.nn.sigmoid(self.output, name='scores')

        # Define loss function
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.input_y, name='losses')
            self.loss = tf.reduce_mean(losses + regular_const * self.regularizer_output)

# Model parameters
tf.flags.DEFINE_string('data_path', "Data/database_DM_reduced_category.csv", 'Data file path')
tf.flags.DEFINE_integer('num_training', 73800, "Number of training samples")
tf.flags.DEFINE_integer('word_embedding_dimension', 300, "Dimension of word embeddings(default: 300)")
tf.flags.DEFINE_integer('filter_size', 3, "length of filter")
tf.flags.DEFINE_integer('num_filters', 128, "Number of filters per filter size(default: 128)")
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, "Dropout keep probability(default: 0.5)")
tf.flags.DEFINE_integer('num_units', 200, "Number of filters per lstm cell(default: 128)")

Flags = tf.flags.FLAGS
Flags._parse_flags()
# Data pre-processing
# ---------------------------------------------------------------------
# Load data
print("Loading data ...")
[data_x, data_y, ids] = load_data(Flags.data_path)
# Padding sentences
print("Padding sentences ...")
data_x_padded = padding(data_x)
# Build vocabulary
print("Building vocabulary ...")
[vocabulary, inverse_vocabulary] = build_vocabulary(data_x_padded)
# Map input sentences and labels with vectors
[data_x_padded_vector, data_y_vector] = map_input_vectors(data_x_padded, data_y, vocabulary)
# Build word embeddings
print("Building word embeddings ...")
word2vec = build_word2vec("Data/GoogleNews-vectors-negative300.bin", vocabulary, inverse_vocabulary)
word_embeddings = build_word_embedding(word2vec, inverse_vocabulary)
# Split training and testing set
data_x_train = data_x_padded_vector[:Flags.num_training]
data_x_test = data_x_padded_vector[Flags.num_training:]
data_y_train = data_y_vector[:Flags.num_training]
data_y_test = data_y_vector[Flags.num_training:]
print(len(data_x_padded[1]), len(data_y[1]))
print("Data is ready!")

# Training

# Define training parameters
epochs = 50
batch_size = 128

# Create session
with tf.Graph().as_default():
    with tf.Session() as sess:
        clstm_nn = Clstm_Text(sentence_length=len(data_x_train[1]),
            num_labels=len(data_y_train[1]),
            embedding_size=Flags.word_embedding_dimension,
            filter_size=Flags.filter_size,
            num_filters=Flags.num_filters,
            num_units=Flags.num_units,
            dropout_keep_prob=0.5,
            regular_const=0.1,

            embedding_matrix=word_embeddings)
        # Define training operation for gradient update
        global_step = tf.Variable(0, name="global_step", trainable=False)  # a counter to count on steps
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(clstm_nn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Generate output directory
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "clstm_text_runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Generate summaries
        loss_summary = tf.summary.scalar("loss", clstm_nn.loss)
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
                clstm_nn.input_x: x_batch,
                clstm_nn.input_y: y_batch,
            }

            #print("shape of y_batch:{}".format(y_batch.shape()))
            _, step, summaries, loss, scores = sess.run(
                [train_op, global_step, train_summary_op, clstm_nn.loss, clstm_nn.scores],
                feed_dict)
            train_summary_writer.add_summary(summaries, step)


        def test_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a test set
            """
            feed_dict = {
                clstm_nn.input_x: x_batch,
                clstm_nn.input_y: y_batch,
            }
            step, summaries, loss, scores = sess.run(
                [global_step, test_summary_op, clstm_nn.loss, clstm_nn.scores],
                feed_dict)
            average_precision = label_ranking_average_precision_score(y_batch, scores)
            ranking_loss = label_ranking_loss(y_batch, scores)
            coverage = coverage_error(y_batch, scores)
            prediction = scores
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            hammingloss = hamming_loss(y_batch, prediction)
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
        batches = batch_iterator(list(zip(data_x_train, data_y_test)), batch_size, epochs)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                print("\nEvaluation:")
                test_step(data_x_test, data_y_test, writer=test_summary_writer)
                print()
            if current_step % 100 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)  # path will be returned
                print("Saved model checkpoint to {}\n".format(path))