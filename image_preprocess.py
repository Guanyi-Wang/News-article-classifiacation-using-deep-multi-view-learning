from os import listdir
from os.path import join
import cv2


def image_reader(dir_path):
    Ids = []
    image = []
    # list of dir names in general path
    dirs = listdir(dir_path)
    for dire in dirs:
        # list of file names in each dir
        files = listdir(join(dir_path, dire))
        for file in files:
            Ids.append(file.strip('.text'))
            path = join(dir_path, dire, file)
            image.append(cv2.imread(path))
    return [Ids, image]


[x,y] = image_reader('Data/DM-images')
print()