ids = []
# read id
with open('Data/DM_ID.txt') as id_file:
    for line in id_file.readlines():
        ids.append(line.rstrip())

# # print(ids)
# # reduce category
# category_dic = {}

#         category_dic[id] = cate
# with open('Data/DM_CATEGORY.txt', 'w', encoding="utf-8") as file:
#     for id in ids:
#         try:
#             file.write('***id:'+id+'***category:'+category_dic[id][1])
#         except KeyError:
#             ids.remove(id)
# print(len(ids))
#
# headline_dic = {}
# with open('Data/DM_RAW_HEADLINE.txt', encoding='utf-8') as text_file:
#     for line in text_file.readlines():
#         id = line.split('***')[1].split('id:')[1]
#         text = line.split('***')[2].split('headline:')
#         headline_dic[id] = text
# with open('Data/DM_HEADLINE.txt','w', encoding='utf-8') as hd_write:
#     for id in ids:
#         try:
#             hd_write.write('***id:'+id+'***headline:'+headline_dic[id][1])
#         except KeyError:
#             ids.remove(id)
# print(len(ids))
# # with open('Data/DM_ID.txt', 'w', encoding='utf-8') as output:
# #     for id in ids:
# #         output.write(id + '\n')
print(len(ids))
text_dic = {}
i = 0
with open('Data/DM_RAW_TEXT.txt', encoding='latin-1') as text_file:
    for line in text_file.readlines():
        try:
            x = line.split('***')[1].split('id:')
            id = line.split('***')[1].split('id:')[1]
            text = line.split('***')[2].split('text:')
            text_dic[id] = text
        except IndexError:
            i = i+1
            print("!!!!"+str(line)+str(i))
with open('Data/DM_TEXT.txt', 'w', encoding='latin-1') as hd_write:
    for id in ids:
        try:
            hd_write.write('***id:'+id+'***headline:'+text_dic[id][1])
        except KeyError:
            i = i+1
            print(i)
            ids.remove(id)
print(len(ids))
print(len(text_dic))
with open('Data/DM_ID.txt', 'w', encoding='utf-8') as output:
    for id in ids:
        output.write(id + '\n')
print(len(ids))