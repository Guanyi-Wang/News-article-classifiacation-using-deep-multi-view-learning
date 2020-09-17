from read_json import read_jason
from text_preprocess import text_mining_headline

path_list = ['Data/DM_json/DM_Partial_0.jsonld', 'Data/DM_json/DM_Partial_1.jsonld', 'Data/DM_json/DM_Partial_2.jsonld',
             'Data/DM_json/DM_Partial_3.jsonld', 'Data/DM_json/DM_Partial_4.jsonld', 'Data/DM_json/DM_Partial_5.jsonld',
             'Data/DM_json/DM_Partial_6.jsonld', 'Data/DM_json/DM_Partial_7.jsonld', 'Data/DM_json/DM_Partial_8.jsonld',
             'Data/DM_json/DM_Partial_9.jsonld', 'Data/DM_json/DM_Partial_10.jsonld']
ids = []
headlines = []
categories = []
for path in path_list:
    id, hd, im, cat = read_jason(path)
    ids = ids + id
    headlines = headlines + hd
    categories = categories + cat

headlines = text_mining_headline(headlines)
with open('Data/DM_RAW_HEADLINE.txt', 'w', encoding="utf8") as file:
    for ID, all_cate in zip(ids, headlines):
        file.write('***id:' + ID + '***headline:' + ' '.join(all_cate) + '\n')