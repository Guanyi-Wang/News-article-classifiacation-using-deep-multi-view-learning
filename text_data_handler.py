"""
This script provide functions to generate input and output data for both training and testing.

"""
# import
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import *
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim


def load_data(db_path):

    # read data from database
    data_x = []  # input data for both training and testing
    data_y = []  # output labels for both training and testing
    ids = []  # list of ids
    with open(db_path, 'r') as db:
        for line in db.readlines():
            line = line.split('*--*')
            # get id
            id_num = line[1].split('id: ')[1]
            # get category and text
            try:
                category = line[2].split('category: ')[1].rstrip()
                text = line[3].split('text: ')[1].rstrip()
            except:
                print(id_num)
            data_y.append(set(category.split()))
            data_x.append(text.split())
            ids.append(id_num)

    # generate binary multi-label for output data
    mlb = MultiLabelBinarizer()
    data_y_binary = mlb.fit_transform(data_y)
    num_category = len(data_y_binary[0])
    return [data_x, data_y_binary, ids]


def padding(sentences, padding_word="</s>"):
    """

    :param sentences:
    :param padding_word:
    :return:padded sentences
    """
    sentence_length = max(len(row) for row in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        temp_sentence = sentences[i] + (sentence_length-len(sentences[i]))*[padding_word]
        padded_sentences.append(temp_sentence)
    return padded_sentences

def build_vocabulary(sentences):
    """
    Build vocabulary and inverse vocabulary based on word count in input sentences.
    :param sentences: Input sentences
    :return: vocabulary and inverse vocabulary:
    """
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]  # a list of strings ordered by frequency
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}  # vocabulary is a dictionary {'<s>':0,'<year>':1} ordered
    #  by frequency
    print("{}={}".format('Vocabulary size', len(vocabulary)))
    return [vocabulary, vocabulary_inv]


def tfidf_matrix(sentences):
    tfidf_vec = TfidfVectorizer(smooth_idf=False)
    tfidf = tfidf_vec.fit_transform(sentences)
    return tfidf.toarray()


def build_word2vec(file, vocabulary, vocabulary_inverse):
    """
    :param file: pre-trained word2vec C format file
    :param vocabulary: The vocabulary dictionary get from your sentences
    :param vocabulary_inverse: The inverse vocabulary list get from your sentences
    :return: A dictionary of words , use pre-trained word2vec if exists and random vectors if not
    """
    word2vec = {}
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)
    # word_vectors = gensim.models.Word2Vec.load(file)
    print("Word2vec file loaded.")
    # Load words exist in pre-trained word2vec
    for word in vocabulary:
        if word in word_vectors:
            word2vec[word] = word_vectors[word]
    # Generate random vector for missing words
    for word in vocabulary_inverse:
        if word not in word_vectors:
            word2vec[word] = np.random.uniform(-0.25, 0.25, 300)  # 0.25 is chosen so the unknown
            # vectors have (approximately) same variance as pre-trained ones
    return word2vec


def build_word_embedding(word2vec, vocabulary_inverse):
    word_embedding = []
    for word in vocabulary_inverse:
        word_embedding.append(word2vec[word])
    return np.array(word_embedding, np.float32)


def map_input_vectors(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return x, y


def batch_iterator(data, batch_size, num_epoch):
    sample = np.array(data)
    sample_size = len(sample)
    num_batch_per_epoch = int(sample_size/batch_size) + 1
    for i in range(num_epoch):
        # randomly shuffle the sample
        shuffled_index = np.random.permutation(sample_size)
        shuffled_sample = sample[shuffled_index]
        # generate batches and return iterator for each batch
        for num_batch in range(num_batch_per_epoch):
            start = num_batch * batch_size
            end = min((num_batch + 1)*batch_size, sample_size)
            if start >= end:
                break
            yield shuffled_sample[start: end]
# data_x_train_padded = padding(data_x_train)
# [voc, voc_inv] = build_vocabulary(data_x_train_padded)
# word2vec = build_word2vec("Data/GoogleNews-vectors-negative300.bin", voc, voc_inv)
# word_embedding = build_word_embedding(word2vec, voc_inv)
# print(word_embedding)

def load_data_new(p1, p2):
    # read data from database
    data_x = []  # input data for both training and testing
    data_y = []  # output labels for both training and testing
    ids = []  # list of ids
    with open(p1, 'r', encoding='utf-8') as db:
        for line in db.readlines():
            line = line.split('***')
            hd = line[2].split('headline:')[1].rstrip().split()
            data_x.append(hd)
    with open(p2, 'r', encoding='utf-8') as db:
        for line in db.readlines():
            line = line.split('***')
            hd = line[2].split('category:')[1].rstrip().split()
            data_y.append(hd)
    # generate binary multi-label for output data
    mlb = MultiLabelBinarizer()
    data_y_binary = mlb.fit_transform(data_y)
    return [data_x, data_y_binary, ids]
