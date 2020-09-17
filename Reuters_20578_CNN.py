
from text_data_handler import *
from xml_cnn import *
import os
import time
from Reuters_21578_data_handler import get_data_for_CNN
from Kim_CNN import *

from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss, coverage_error, hamming_loss, \
    f1_score, precision_score, recall_score

# Define Parameters
# -----------------------------------------------------------------
# Data loading
tf.flags.DEFINE_string('data_path', "Data/database_DM_reduced_category.csv", 'Data file path')
tf.flags.DEFINE_integer('num_training', 7769, "Number of training samples(default: 7769)")

# Model parameters
tf.flags.DEFINE_integer('word_embedding_dimension', 300, "Dimension of word embeddings(default: 300)")
tf.flags.DEFINE_string("filter_sizes", "2,4,8", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer('num_filters', 128, "Number of filters per filter size(default: 128)")
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, "Dropout keep probability(default: 0.5)")
tf.flags.DEFINE_integer("num_hidden", 50, "Number of neurons in hidden layers ")
tf.flags.DEFINE_integer("regular_const", 0.01, "Regularization constant")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 8, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on test set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("chunk_size", 2, "Chunk size during dynamic maxpooling (default: 2)")

# Print parameters
Flags = tf.flags.FLAGS
Flags._parse_flags()
print("\nParameters:")
for attr, value in sorted(Flags.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("Start data pre-processing")
# ---------------------------------------------------------------------

# Data pre-processing
# ---------------------------------------------------------------------
# Load data
print("Loading data ...")
[data_x, data_y] = get_data_for_CNN()
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
# word2vec = build_word2vec("Data/DM_HEADLINE_model", vocabulary, inverse_vocabulary)
word_embeddings = build_word_embedding(word2vec, inverse_vocabulary)
# Split training and testing set
data_x_train = data_x_padded_vector[:Flags.num_training]
data_x_test = data_x_padded_vector[Flags.num_training:]
data_y_train = data_y_vector[:Flags.num_training]
data_y_test = data_y_vector[Flags.num_training:]
print(len(data_x_padded[1]), len(data_y[1]))
print("Data is ready!")
# -----------------------------------------------------------------

# Training
# -----------------------------------------------------------------
with tf.Graph().as_default():
    session_cof = tf.ConfigProto(
        allow_soft_placement=True,  # allows TensorFlow to fall back on a device with a certain operation implemented
        # when the preferred device doesn’t exist
        log_device_placement=False  # log on which devices (CPU or GPU) it places operations
    )
    sess = tf.Session(config=session_cof)
    with sess.as_default():
        text_cnn = XML_Cnn(sentence_length=len(data_x_train[1]),
                           num_labels=len(data_y_train[1]),
                           embedding_size=Flags.word_embedding_dimension,
                           filter_sizes=list(map(int, Flags.filter_sizes.split(","))),
                           num_filters=Flags.num_filters,
                           chunk_size=Flags.chunk_size,
                           num_hidden_neuron=Flags.num_hidden,
                           embedding_matrix=word_embeddings,
                           regular_const=Flags.regular_const)
        #     sentence_length=len(data_x_train[1]),
        #     num_labels=len(data_y_train[1]),
        #     embedding_size=Flags.word_embedding_dimension,
        #     filter_sizes=list(map(int, Flags.filter_sizes.split(","))),
        #     num_filters=Flags.num_filters,
        #     embedding_matrix=word_embeddings,
        #     num_hidden_neuron=Flags.num_hidden,
        #     chunk_size=Flags.chunk_size,
        #     regular_const=Flags.regular_const)

    # Define training operation for gradient update
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(text_cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Generate output directory
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "text_cnn_runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Generate summaries
    loss_summary = tf.summary.scalar("loss", text_cnn.loss)
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


    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            text_cnn.input_x: x_batch,
            text_cnn.input_y: y_batch,
            text_cnn.dropout_keep_prob: Flags.dropout_keep_prob
        }
        _, step, summaries, loss, scores = sess.run(
            [train_op, global_step, train_summary_op, text_cnn.loss, text_cnn.scores],
            feed_dict)
        train_summary_writer.add_summary(summaries, step)


    def test_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a test set
        """
        feed_dict = {
            text_cnn.input_x: x_batch,
            text_cnn.input_y: y_batch,
            text_cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, scores = sess.run(
            [global_step, test_summary_op, text_cnn.loss, text_cnn.scores],
            feed_dict)
        average_precision = label_ranking_average_precision_score(y_batch, scores)
        ranking_loss = label_ranking_loss(y_batch, scores)
        coverage = coverage_error(y_batch, scores)
        prediction = scores
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0
        hammingloss = hamming_loss(y_batch, prediction)
        micro_precision = precision_score(y_batch, prediction, average='micro')
        macro_precision = precision_score(y_batch, prediction, average='macro')
        micro_recall = recall_score(y_batch, prediction, average='micro')
        macro_recall = recall_score(y_batch, prediction, average='macro')
        micro_f1 = f1_score(y_batch, prediction, average='micro')
        macro_f1 = f1_score(y_batch, prediction, average='macro')

        print("{}: step {}, loss {:g}, AP {:g}, ranking_loss {:g}, coverage {:g}, hamming_loss {:g}, "
              "micro_precision {:g}, macro_precision {:g}, micro_recall {:g}, macro_recall {:g}, micro_f1 {:g}, "
              "macro_f1 {:g}".format('dev', step, loss, average_precision, ranking_loss, coverage, hammingloss, micro_precision,
                                     macro_precision, micro_recall, macro_recall, micro_f1, macro_f1))
        if writer:
            writer.add_summary(summaries, step)

    # Generate batches
    batches = batch_iterator(
        list(zip(data_x_train, data_y_train)), Flags.batch_size, Flags.num_epochs)
    # Training loop. For each batch...

    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % Flags.evaluate_every == 0:
            print("\nEvaluation:")
            test_step(data_x_test, data_y_test, writer=test_summary_writer)
            print()
        if current_step % Flags.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)  # path will be returned
            print("Saved model checkpoint to {}\n".format(path))
