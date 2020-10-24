# masquito-mask-detection

## overview

Machine learning part of a fullstack application to tell if you're wearing a mask right

Followed ImageAI's [prediction model](https://ImageAI.readthedocs.io/en/latest/custom/)

Used images from [Kaggle](https://www.kaggle.com/ivandanilovich/medical-masks-dataset-images-tfrecords)

## mmds_to_imageai.py

Converts the dataset from Kaggle's format to ImageAI's format

```
medical-masks-dataset/
    images/
        *.jpg
    labels/
        *.xml
```

to

```
training_data/
    cropped/
        mask/
        none/
        poor/
```

## train_model.py

Trains a simple prediction model based on ImageAI's Prediction class

## test_model.py

Test the model against real-world images

To try it out yourself, download `model_*.h5` from the releases tab and place it in `training_data/models/`. Then modify this script's `model` variable to match its filename.

## training_data/ structure

```
training_data/
    json/
        model_class.json
    logs/
    models/
    test/
        mask/
        none/
        poor/
    train/
        mask/
        none/
        poor/
    cropped/
        mask/
        none/
        poor/
```

## pip dependencies

```
tensorflow<2
scipy<1.5
numpy
keras
opencv-python
pillow
```
