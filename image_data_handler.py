import h5py
import json
import numpy as np
from text_data_handler import load_data
def load_image_data():
    with h5py.File('Data/image_features/dm_feats/fc7_vgg_feats_hdf5.mat', 'r') as file:
        data = file.get('feats')
        values = data.value
        # print(values.shape)

    # List to store image id
    id_image =[]
    with open('Data/image_features/dm_feats/output') as id_file:
        for line in id_file.readlines():
            line = line.split('/')
            id_image.append(line[6].split('.jpg')[0])

    # Generate image dictionary using image id and value
    image_dic = dict(zip(id_image, values))

    # Get category and id list from database
    images = []
    [x, category, ids] = load_data('Data/database_DM_reduced_category.csv')

    for ID in ids:
        # Find images category and store it in a list
        image = image_dic.get(ID)
        # for i in image:
        #     if(i.dtype!='float32'):
        #         i = i.rstrip()
        images.append(image)
    return[np.array(images), np.array(category)]
#
# # Read data from jsonld file
# with open("Data/DM_json/DM_Partial_0.jsonld", encoding="utf8") as js_file:
#     js_data = json.load(js_file)
#
# reverse = js_data['@reverse']
# publisher_list = reverse['publisher']

# ID = []
# date = []
# headline = []
# url = []
# image = []
# category = []
# about = []
#
# for publisher in publisher_list:
#     # Handling missing values
#     publisher.setdefault('image', 'None')
#     publisher.setdefault('category', 'None')
#     publisher.setdefault('about', 'None')
#
# for publisher in publisher_list:
#     if(publisher['image'] != 'None')and(publisher['category'] != 'None'):
#         ID.append(publisher['@id'])
#         date.append(publisher['datePublished'])
#         headline.append(publisher['headline'])
#         url.append(publisher['url'])
#         image.append(publisher['image'])
#         category.append(publisher['category'])
#         about.append(publisher['about'])
#
#
# print(len(ID))

