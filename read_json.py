import json


def read_jason(path):
    with open(path, encoding="utf8") as js_file:
        js_data = json.load(js_file)

    reverse = js_data['@reverse']
    publisher_list = reverse['publisher']
    ID = []
    date = []
    headline = []
    url = []
    image = []
    category = []

    for publisher in publisher_list:

        publisher.setdefault('image', 'None')  # Handle missing values as None
        publisher.setdefault('category', 'None')  # Handle missing values as None
        publisher.setdefault('about', 'None')  # Handle missing values as None

    for publisher in publisher_list:
        if publisher['category']!='None':
            ID.append(publisher['@id'])
            date.append(publisher['datePublished'])
            headline.append(publisher['headline'])
            url.append(publisher['url'])
            image.append(publisher['image'])
            category.append(publisher['category'])
    return [ID, headline, image, category]

