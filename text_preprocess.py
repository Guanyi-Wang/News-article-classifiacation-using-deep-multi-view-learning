from os import listdir
from os.path import join
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import string
from collections import Counter


def text_reader(dir_path):
    Ids = []
    text = []
    # list of dir names in general path
    dirs = listdir(dir_path)
    for dire in dirs:
        # list of file names in each dir
        files = listdir(join(dir_path, dire))
        for file in files:
            Ids.append(file.strip('.text'))
            path = join(dir_path, dire, file)
            with open(path, 'r', encoding='latin-1') as f:
                text.append(f.readlines()[0])  # [0] convert list to string
    return [Ids, text]


def text_writer(x, y, dir, encoding):
    with open(dir, 'w', encoding=encoding) as file:
        for i, j in zip(x, y):
            file.write('***id:'+i+'***text:'+j+'\n')


def text_mining(text, min_frequency):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    outputs = []
    word_counter = Counter()
    frequency_filter = []
    i = 0
    for t in text:  # t is a string
        i = i+1
        tr = str.maketrans("", "", string.punctuation)  # a translation map
        t = t.translate(tr)  # remove punctuation
        tokenized = word_tokenize(t)  # list of string
        # print(tokenized)
        stemmed = [stemmer.stem(i) for i in tokenized]  # list of string
        # print(stemmed)
        without_stopwords = [j.lower() for j in stemmed if j.lower() not in stop_words]
        # print(without_stopwords)
        outputs.append(without_stopwords)
        word_counter.update(without_stopwords)
        print(i)
    print('finish one')
    i = 0
    for item in word_counter.items():
        i = i +1
        if item[1] < min_frequency:  # frequency less than 3
            frequency_filter.append(item[0])  # add the word into filter
        print(i)
    filtered_outputs = []
    i = 0
    for output in outputs:
        i = i+1
        filtered = [f for f in output if f not in frequency_filter]
        filtered_outputs.append(filtered)
        print(i)
    return filtered_outputs


def text_mining_headline(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    outputs = []
    for t in text:  # t is a string
        tr = str.maketrans("", "", string.punctuation)  # a translation map
        t = t.translate(tr)  # remove punctuation
        tokenized = word_tokenize(t)  # list of string
        # print(tokenized)
        stemmed = [stemmer.stem(i) for i in tokenized]  # list of string
        # print(stemmed)
        without_stopwords = [j.lower() for j in stemmed if j.lower() not in stop_words]
        # print(without_stopwords)
        outputs.append(without_stopwords)
    return outputs


def write_text(data_path, min_frequency, output_dir):
    [ids, texts] = text_reader(data_path)
    print("Finish reading text and id...")
    texts = text_mining(texts, min_frequency)
    print("Finish text mining...")
    with open(output_dir, 'w', encoding='latin-1') as file:
        for ID, text in zip(ids, texts):
            file.write('***id:'+ID+'***text:'+' '.join(text)+'\n')
    print("Finish writing to file:"+output_dir)


#
# write_text('Data/DM-articles',30,'Data/DM_RAW_TEXT.txt')
#

