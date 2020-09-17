from nltk.corpus import stopwords, reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from Reuters_21578_data_handler import tokenize
from sklearn.multioutput import ClassifierChain
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, label_ranking_loss, average_precision_score, \
    hamming_loss, coverage_error
stop_words = stopwords.words("english")

# List of document ids
documents = reuters.fileids()

train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                            documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                           documents))

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

# Tokenisation
vectorizer = TfidfVectorizer(stop_words=stop_words,
                             tokenizer=tokenize)

# Learn and transform train documents
vectorised_train_documents = vectorizer.fit_transform(train_docs)
vectorised_test_documents = vectorizer.transform(test_docs)

# Transform multilabel labels
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id)
                                  for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id)
                             for doc_id in test_docs_id])
# OneVsRestClassifier wrapper.
ovr = OneVsRestClassifier(LinearSVC(random_state=3))
ovr.fit(vectorised_train_documents, train_labels)
Y_pred_ovr = ovr.predict(vectorised_test_documents)

# Fit an ensemble of logistic regression classifier chains and take the
# take the average prediction of all the chains.
chains = [ClassifierChain(LinearSVC(), order='random', random_state=i)
          for i in range(10)]
for chain in chains:
    chain.fit(vectorised_train_documents, train_labels)

Y_pred_chains = np.array([chain.predict(vectorised_test_documents) for chain in chains])

Y_pred_ensemble = Y_pred_chains.mean(axis=0)
prediction = np.where(Y_pred_ensemble < 0.5, 0, Y_pred_ensemble)
prediction = np.where(prediction >= 0.5, 1, prediction)

# Evaluation
average_precision = average_precision_score(test_labels, prediction)
ranking_loss = label_ranking_loss(test_labels, prediction)
coverage = coverage_error(test_labels, prediction)
hamming_loss = hamming_loss(test_labels, prediction)
micro_precision = precision_score(test_labels, prediction, average='micro')
micro_recall = recall_score(test_labels, prediction, average='micro')
micro_f1 = f1_score(test_labels, prediction, average='micro')
macro_precision = precision_score(test_labels, prediction, average='macro')
macro_recall = recall_score(test_labels, prediction, average='macro')
macro_f1 = f1_score(test_labels, prediction, average='macro')

print("{}: AP {:g}, ranking_loss {:g}, coverage {:g}, hamming_loss {:g}, "
              "micro_precision {:g}, macro_precision {:g}, micro_recall {:g}, macro_recall {:g}, micro_f1 {:g}, "
              "macro_f1 {:g}".format('dev', average_precision, ranking_loss, coverage, hamming_loss, micro_precision,
                                     macro_precision, micro_recall, macro_recall, micro_f1, macro_f1))


