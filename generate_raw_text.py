from text_preprocess import *
ids, texts = text_reader('Data/DM-articles')
# for id, text in zip(ids,texts):
#     if id != '' or text !='':
#         ids.remove(id)
#         texts.remove(text)
text_writer(ids, texts, 'Data/DM_RAW_TEXT.txt', 'latin-1')
