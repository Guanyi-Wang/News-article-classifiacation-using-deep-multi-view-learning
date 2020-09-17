from collections import Counter

category_counter = Counter()
ids = []
category = []
with open('Data/DM_RAW_CATEGORY.txt', encoding='utf-8') as cate_file:
    for line in cate_file.readlines():
        id = line.split('***')[1].split('id:')[1]
        cate = line.split('***')[2].split('category:')[1].split()
        category_counter.update(cate)
        ids.append(id)
        category.append(cate)
print(category_counter.most_common(100))
print()