import gensim
def word2vec_trainer(corpus_path, output_path):
    headlines = []
    with open(corpus_path, 'r', encoding='utf-8') as db:
        for line in db.readlines():
            line = line.split('***')
            hd = line[2].split('headline:')[1].rstrip().split()
            headlines.append(hd)
    model = gensim.models.Word2Vec(headlines, min_count=10, size=200, workers=4)
    model.save(output_path)
    return model
model = word2vec_trainer('Data/DM_HEADLINE.txt', 'Data/DM_HEADLINE_model')
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))