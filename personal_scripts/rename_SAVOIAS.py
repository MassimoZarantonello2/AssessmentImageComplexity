import os

path = './Savoias-Dataset-master/Images'
categories = os.listdir(path)

for category in categories:
    images = os.listdir(os.path.join(path, category))
    for image in images:
        os.rename(os.path.join(path, category, image), os.path.join(path, category + '_' + image))