from read_json import read_jason
# Get raw category from jsonld file
path_list = ['Data/IND_json/IND_Partial_0.jsonld', 'Data/IND_json/IND_Partial_1.jsonld', 'Data/IND_json/IND_Partial_2.jsonld',
             'Data/IND_json/IND_Partial_3.jsonld']
ids = []
categories = []
for path in path_list:
    id, hd, im, cat = read_jason(path)
    ids = ids + id
    categories = categories + cat

all_category = []
for news in categories:
    category = []
    for cate in news:
        cate = cate.split('http://en.wikipedia.org/wiki/Category:')[1]
        category.append(cate)
    all_category.append(category)

with open('Data/IND_RAW_CATEGORY.txt', 'w', encoding="utf8") as file:
    for ID, all_cate in zip(ids, all_category):
        file.write('***id:'+ID+'***category:'+' '.join(all_cate)+'\n')
# print(all_category)

