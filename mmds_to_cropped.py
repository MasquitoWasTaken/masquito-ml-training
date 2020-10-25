import os
from PIL import Image
import xml.etree.ElementTree as ET

path = './medical-masks-dataset/medical-masks-dataset/'
save = './cropped/'

# Create directories to contain cropped faces
categories = ['mask', 'none', 'poor']
for category in categories:
    if not os.path.exists(save + category):
        os.makedirs(save + category)

# For each original image, find the face bounds
label_files = os.listdir(path + 'labels/')
for label_file in label_files:
    tree = ET.parse(path + 'labels/' + label_file)
    root = tree.getroot()
    filename = root.find('./filename').text
    image = Image.open(path + 'images/' + filename)

    # For each face, save the face as a separate file
    for i, item in enumerate(root.findall('./object')):
        value = item.find('name').text
        dimensions = map(lambda x: int(x), (
            item.find('./bndbox/xmin').text,
            item.find('./bndbox/ymin').text,
            item.find('./bndbox/xmax').text,
            item.find('./bndbox/ymax').text
        ))

        # Sometimes the dataset has bad parameters; ignore the invalid data
        try:
            cropped = image.crop(dimensions)
            cropped.save(save + value + '/' + filename +
                         '-' + str(i) + '.jpg')
        except:
            pass
