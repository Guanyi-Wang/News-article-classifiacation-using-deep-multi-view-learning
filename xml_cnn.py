import tensorflow as tf
from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss, coverage_error, hamming_loss


class XML_Cnn(object):

    def __init__(self, sentence_length, num_labels, embedding_size, filter_sizes, num_filters, chunk_size, num_hidden_neuron, regular_const,
                 embedding_matrix=[]):

        # Placeholders for input output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_labels], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Embedding Layer
        with tf.device('/cpu:0'), tf.name_scope('embedding_layer'):  # force to run on CPU
            word_embedding = tf.Variable(embedding_matrix, trainable='False',
                                         name='embedding_matrix')  # embedding matrix
            # real embedding operation, returns a 3-dimensional tensor:[None, sentence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(word_embedding, self.input_x)
            # expand 1 dimension(channel) :[batch:None, width:sentence_length, height:embedding_size, 1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)   # expand 1 dimension at the last

        # Convolution and Max-Pooling Layers
        pooled_outputs = []
        total_num_features = 0
        for i, filter_size in enumerate(filter_sizes):  # iterate on each filter_size
            with tf.name_scope('conv_maxpool_%s' % filter_size):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                filter_matrix = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1,), trainable=True,
                                            name='filter_matrix')   # filter matrix
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")   # bias term
                conv = tf.nn.conv2d(self.embedded_chars_expanded, filter_matrix, strides=[1, 1, 1, 1], padding='VALID',
                                    name='convolution')
                # Apply nonlinear function ReLu(rectified linear unit)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='ReLu')
                # Max-pooling over each feature
                pooled = tf.nn.max_pool(h, ksize=[1, chunk_size, 1, 1], strides=[1, 1, 1, 1], padding='VALID',
                                        name="Max_pooling")  # [?, sentence_length-filter_size+1-chunk_size+1,1,num_filters]
                # num_features_per_filter = sentence_length-filter_size+1-chunk_size+1
                # num_features = num_features_per_filter*num_filters
                # total_num_features = total_num_features+num_features
                # pooled = tf.reshape(pooled, [-1, num_features])
                # flatten
                pooled_flattened = tf.contrib.layers.flatten(pooled)
                pooled_outputs.append(pooled_flattened)
                # pooled_outputs = tf.concat([pooled, pooled_outputs], 1)
        # Combine all the features
        self.h_pooled_flat = tf.concat(pooled_outputs, 1)  # flatten the dimension when possible
        total_num_features = self.h_pooled_flat.get_shape().as_list()[1]
        # Add dropout layer
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pooled_flat, self.dropout_keep_prob)

        # Add hidden bottleneck layer
        with tf.name_scope("hidden_bottleneck_layer"):
            W_hidden = tf.get_variable(
                "W_hidden",
                shape=[total_num_features, num_hidden_neuron],
                initializer=tf.contrib.layers.xavier_initializer())  # this initializer is designed to keep the scale
            # of the gradients roughly the same in all layers
            b_hidden = tf.Variable(tf.constant(0.1, shape=[num_hidden_neuron]), name="b_hidden")
            self.regularizer_hidden = tf.nn.l2_loss(W_hidden, name="hidden_layer_regularizer")
            self.hidden_output = tf.nn.sigmoid(tf.nn.xw_plus_b(self.h_drop, W_hidden, b_hidden, name="hidden_layer_output"))
        # scores and predictions
        with tf.name_scope("output"):
            W_output = tf.get_variable(
                "W_output",
                shape=[num_hidden_neuron, num_labels],
                initializer=tf.contrib.layers.xavier_initializer())  # this initializer is designed to keep the scale
            # of the gradients roughly the same in all layers
            b_output = tf.Variable(tf.constant(0.1, shape=[num_labels]), name="b_output")
            self.regularizer_output = tf.nn.l2_loss(W_output, name="output_layer_regularizer")
            self.output = tf.nn.xw_plus_b(self.hidden_output, W_output, b_output, name="output")
            self.scores = tf.nn.sigmoid(self.output, name='scores')

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.input_y)
            self.loss = tf.reduce_mean(losses + regular_const * self.regularizer_hidden +
                                       regular_const * self.regularizer_output)
        # # Calculate average precision
        # with tf.name_scope('average_precision'):
        #     self.average_precision = label_ranking_average_precision_score(self.scores, self.scores)

        # # Calculate ranking loss
        # with tf.name_scope('ranking_loss'):
        #     self.ranking_loss = label_ranking_loss(self.input_y, self.scores)
        #
        # # Calculate coverage
        # with tf.name_scope('coverage'):
        #     self.coverage = coverage_error(self.input_y, self.scores)





