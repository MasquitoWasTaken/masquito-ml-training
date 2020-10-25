import os
from PIL import Image

path = './cropped/'
categories = ['mask', 'none', 'poor', 'with_mask',
              'without_mask', 'mask_weared_incorrect']

# Try to open every image with PIL; verify that each image is openable
for category in categories:
    files = os.listdir(path + category)
    for filename in files:
        try:
            image = Image.open(path + category + '/' + filename)
        except:
            print(filename)
            os.remove(path + category + '/' + filename)
