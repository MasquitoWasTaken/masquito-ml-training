# masquito-mask-detection

## overview

Machine learning part of a fullstack application to tell if you're wearing a mask right

Followed ImageAI's [prediction model](https://ImageAI.readthedocs.io/en/latest/custom/)

Used images from Kaggle:

-   [Medical Masks Dataset](https://www.kaggle.com/ivandanilovich/medical-masks-dataset-images-tfrecords)
-   [Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection)

## run this program

1. Clone this repository
2. Install the pip dependencies listed below
3. Download the latest model from the [releases](https://github.com/MasquitoWasTaken/masquito-ml-training/releases) tab
4. Place the `.h5` file in `training_data/models/`
5. Edit `test_model.py`'s `model` variable to match the filename
6. Place a test image (or use an included one) in `test_images/`
7. Edit `test_model.py`'s `test_image` variable to match the filename
8. Run the code with `python ./test_model.py`

## dataset construction

We used two publicly available datasets from the website Kaggle. Two scripts (`mmds_to_cropped.py` and `fmdds_to_cropped.py`) cropped out the faces labelled in each dataset. `check_images.py` made sure there was no corrupt data. Then we combined the processed images for both datasets into `aggregate/`. Finally, we manually cut down each class into 270 images (limited in quantity by the `improper` class) and used 220:50 train:test ratio, formatted in `training_data/` using ImageAI's directory structure.

## scripts

### mmds_to_imageai.py and fmmds_to_cropped.py

Converts the dataset from Kaggle's format to ImageAI's format

### check_images.py

For an unknown reason, Pillow's `Image.save` function in `mmds_to_imageai.py` occasionally spits out unreadable data -- data that can't be parsed by `Image.open`. For that reason, this script finds the invalid files and deletes them. Bye-bye!

### train_model.py

Trains a simple prediction model based on ImageAI's Prediction class

### test_model.py

Tests the model against real-world images

## directory structure

```
face-mask-detection-dataset/
    annotations/
    images/
medical-masks-dataset/
    medical-masks-dataset/
        labels/
        images/
cropped/
    mask/
    mask_weared_incorrect/
    none/
    poor/
    with_mask/
    without_mask/
aggregate/
    mask/
    improper/
    none/
training_data/
    json/
        model_class.json
    logs/
    models/
    test/
        mask/
        none/
        improper/
    train/
        mask/
        none/
        improper/
```

## pip dependencies

Note: you **must** use Python <3.8 (I recommend 3.7).

```
tensorflow<2
scipy<1.5
numpy
keras
opencv-python
pillow
```
