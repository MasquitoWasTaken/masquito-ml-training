import os
from PIL import Image
import xml.etree.ElementTree as ET

path = './face-mask-detection-dataset/'
save = './cropped/'

# Create directories to contain cropped faces
categories = ['with_mask', 'without_mask', 'mask_weared_incorrect']
for category in categories:
    if not os.path.exists(save + category):
        os.makedirs(save + category)

# For each original image, find the face bounds
label_files = os.listdir(path + 'annotations/')
for label_file in label_files:
    tree = ET.parse(path + 'annotations/' + label_file)
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
                         '-' + str(i) + '.png')
        except:
            pass
