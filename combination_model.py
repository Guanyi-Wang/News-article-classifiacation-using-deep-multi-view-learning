import tensorflow as tf
from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss, coverage_error, hamming_loss
from text_data_handler import *
import os
import time
import datetime
from print_all_variables import print_all_variables
import image_data_handler


class CombinationModel(object):
    def __init__(self, sentence_length, num_labels, embedding_size, filter_sizes, num_filters, num_features,
                 num_hidden_neuron_text, num_hidden_neuron_image, regular_const, embedding_matrix=[]):

        # Placeholders for input output and dropout
        self.input_x_text = tf.placeholder(tf.int32, [None, sentence_length], name='input_x_text')
        self.input_x_image = tf.placeholder(tf.float32, [None, num_features], name='input_x_image')
        self.input_y = tf.placeholder(tf.float32, [None, num_labels], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Text cnn layer
        with tf.name_scope('text_cnn'):
            # Embedding Layer
            with tf.device('/cpu:0'), tf.name_scope('embedding_layer'):  # force to run on CPU
                word_embedding = tf.Variable(embedding_matrix, trainable='False',
                                             name='embedding_matrix')  # embedding matrix
                # real embedding operation, returns a 3-dimensional tensor:[None, sentence_length, embedding_size]
                self.embedded_chars = tf.nn.embedding_lookup(word_embedding, self.input_x_text)
                # expand 1 dimension(channel) :[batch:None, width:sentence_length, height:embedding_size, 1]
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)   # expand 1 dimension at the last

            # Convolution and Max-Pooling Layers
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):  # iterate on each filter_size
                with tf.name_scope('conv_maxpool_%s' % filter_size):
                    # Convolution layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    filter_matrix = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                                name='filter_matrix')   # filter matrix
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")   # bias term
                    conv = tf.nn.conv2d(self.embedded_chars_expanded, filter_matrix, strides=[1, 1, 1, 1], padding='VALID',
                                        name='convolution')
                    # Apply nonlinear function ReLu(rectified linear unit)
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='ReLu')
                    # Max-pooling over each feature
                    pooled = tf.nn.max_pool(h, ksize=[1, sentence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                            padding='VALID', name="Max_pooling")
                    pooled_outputs.append(pooled)

            # Combine all the features
            total_num_filter = num_filters * len(filter_sizes)
            self.h_pooled = tf.concat(pooled_outputs, 3)
            self.h_pooled_flat = tf.reshape(self.h_pooled, [-1, total_num_filter])  # flatten the dimension when possible

            # Add dropout layer
            with tf.name_scope("dropout_text"):
                self.dropout_text = tf.nn.dropout(self.h_pooled_flat, self.dropout_keep_prob)

            # Add hidden bottleneck layer
            with tf.name_scope("hidden_layer_text"):
                W_hidden_text = tf.get_variable(
                    "W_hidden_text",
                    shape=[total_num_filter, num_hidden_neuron_text],
                    initializer=tf.contrib.layers.xavier_initializer())  # this initializer is designed to keep the scale
                # of the gradients roughly the same in all layers
                b_hidden_text = tf.Variable(tf.constant(0.1, shape=[num_hidden_neuron_text]), name="b_hidden_text")
                self.regularizer_hidden_text = tf.nn.l2_loss(W_hidden_text, name="hidden_layer_regularizer_text")
                self.hidden_output_text = tf.nn.sigmoid(tf.nn.xw_plus_b(self.dropout_text, W_hidden_text, b_hidden_text, name="hidden_layer_output_text"))

        # Image nn layer
        with tf.name_scope('image_nn'):
            # Add dropout layer
            with tf.name_scope("dropout_image"):
                self.dropout_image = tf.nn.dropout(self.input_x_image, self.dropout_keep_prob)

            with tf.name_scope("hidden_layer_image"):
                W_hidden_image = tf.Variable(tf.truncated_normal([num_features, num_hidden_neuron_image], stddev=0.1),
                                             name='W_hidden_image')
                b_hidden_image = tf.Variable(tf.truncated_normal([num_hidden_neuron_image]), name="b_hidden_image")
                self.regularizer_hidden_image = tf.nn.l2_loss(W_hidden_image, name="regularizer_hidden_image")
                self.output_hidden_image = tf.nn.xw_plus_b(self.input_x_image, W_hidden_image, b_hidden_image,
                                                           name='output_hidden')

        # Concatenate layer
        with tf.name_scope('concat'):
            self.concat = tf.concat([self.hidden_output_text, self.output_hidden_image], 1)

        # output layer
        with tf.name_scope('output'):
            W_output = tf.Variable(tf.truncated_normal([num_hidden_neuron_text+num_hidden_neuron_image, num_labels],
                                                       stddev=0.1), name='W_output')
            b_output = tf.Variable(tf.truncated_normal([num_labels]), name="b_output")
            self.regularizer_output = tf.nn.l2_loss(W_output, name="regularizer_output")
            self.output = tf.nn.xw_plus_b(self.concat, W_output, b_output, name='output')
            self.scores = tf.nn.sigmoid(self.output, name='scores')

        # Define loss function
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.input_y, name='losses')
            self.loss = tf.reduce_mean(losses + regular_const * self.regularizer_hidden_text + regular_const *
                                       self.regularizer_hidden_image + regular_const * self.regularizer_output)
# ----------------------------------------------------------------------------------------------------------------------
# Model parameters
tf.flags.DEFINE_string('data_path', "Data/database_DM_reduced_category.csv", 'Data file path')
tf.flags.DEFINE_integer('num_training', 73800, "Number of training samples")
tf.flags.DEFINE_integer('word_embedding_dimension', 300, "Dimension of word embeddings(default: 300)")
tf.flags.DEFINE_integer('filter_sizes', [3, 4, 5], "length of filter")
tf.flags.DEFINE_integer('num_filters', 128, "Number of filters per filter size(default: 128)")
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, "Dropout keep probability(default: 0.5)")
tf.flags.DEFINE_integer('num_units', 200, "Number of filters per lstm cell(default: 128)")
tf.flags.DEFINE_integer('num_hidden_neuron_text', 50, "Number of hidden neuron text hidden layer")
tf.flags.DEFINE_integer('num_hidden_neuron_image', 150, "Number of hidden neuron image hidden layer")
tf.flags.DEFINE_integer('num_features', 4096, "Number of input image features")
tf.flags.DEFINE_integer('regular_const', 0.01, "Regularization constant")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 80, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on test set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")

Flags = tf.flags.FLAGS
Flags._parse_flags()
# ----------------------------------------------------------------------------------------------------------------------
# Data pre-processing
# Load data text data
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
text_x_train = data_x_padded_vector[:Flags.num_training]
text_x_test = data_x_padded_vector[Flags.num_training:]
text_y_train = data_y_vector[:Flags.num_training]
text_y_test = data_y_vector[Flags.num_training:]
print("Text data is ready!")
# Get image data
x, y = image_data_handler.load_image_data()
# Separate training and testing set
image_x_train = x[:Flags.num_training]
image_x_test = x[Flags.num_training:]
image_y_train = y[:Flags.num_training]
image_y_test = y[Flags.num_training:]

with tf.Graph().as_default():
    session_cof = tf.ConfigProto(
        allow_soft_placement=True,  # allows TensorFlow to fall back on a device with a certain operation implemented
        # when the preferred device doesn’t exist
        log_device_placement=False  # log on which devices (CPU or GPU) it places operations
    )
    sess = tf.Session(config=session_cof)
    with sess.as_default():
        comb_model = CombinationModel(
            sentence_length=len(text_x_train[1]),
            num_labels=len(text_y_train[1]),
            embedding_size=Flags.word_embedding_dimension,
            filter_sizes=Flags.filter_sizes,
            num_filters=Flags.num_filters,
            num_features=Flags.num_features,
            embedding_matrix=word_embeddings,
            num_hidden_neuron_text=Flags.num_hidden_neuron_text,
            num_hidden_neuron_image=Flags.num_hidden_neuron_image,
            regular_const=Flags.regular_const)

    # Define training operation for gradient update
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(comb_model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Generate output directory
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "comb_model_runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Generate summaries
    loss_summary = tf.summary.scalar("loss", comb_model.loss)
    # ap_summary = tf.scalar_summary("average_precision", text_cnn.average_precision)
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

    # Initialize all variables
    sess.run(tf.global_variables_initializer())


    def train_step(x_batch_text, x_batch_image, y_batch):
        """
        A single training step
        """
        feed_dict = {
            comb_model.input_x_text: x_batch_text,
            comb_model.input_x_image: x_batch_image,
            comb_model.input_y: y_batch,
            comb_model.dropout_keep_prob: Flags.dropout_keep_prob
        }
        _, step, summaries, loss, scores = sess.run(
            [train_op, global_step, train_summary_op, comb_model.loss, comb_model.scores],
            feed_dict)
        train_summary_writer.add_summary(summaries, step)


    def test_step(x_batch_text, x_batch_image, y_batch, writer=None):
        """
        Evaluates model on a test set
        """
        feed_dict = {
            comb_model.input_x_text: x_batch_text,
            comb_model.input_x_image: x_batch_image,
            comb_model.input_y: y_batch,
            comb_model.dropout_keep_prob: 1.0
        }
        step, summaries, loss, scores = sess.run(
            [global_step, test_summary_op, comb_model.loss, comb_model.scores],
            feed_dict)
        average_precision = label_ranking_average_precision_score(y_batch, scores)
        ranking_loss = label_ranking_loss(y_batch, scores)
        coverage = coverage_error(y_batch, scores)
        prediction = scores
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0
        hammingloss = hamming_loss(y_batch, prediction)
        print("{}: step {}, loss {:g}, AP {:g}, ranking_loss {:g}, coverage {:g}, hamming_loss {:g} ".format('dev',
                                                    step, loss, average_precision, ranking_loss, coverage, hammingloss))
        if writer:
            writer.add_summary(summaries, step)

    # Generate batches
    batches = batch_iterator(
        list(zip(text_x_train, image_x_train, text_y_train)), Flags.batch_size, Flags.num_epochs)
    # Training loop. For each batch...

    for batch in batches:
        x_batch_text, x_batch_image, y_batch = zip(*batch)
        train_step(x_batch_text, x_batch_image, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % Flags.evaluate_every == 0:
            print("\nEvaluation:")
            test_step(text_x_test, image_x_test, text_y_test, writer=test_summary_writer)
            print()
        if current_step % Flags.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)  # path will be returned
            print("Saved model checkpoint to {}\n".format(path))


