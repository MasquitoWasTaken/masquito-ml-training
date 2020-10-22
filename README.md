# masquito-mask-detection

## overview

Machine learning part of a fullstack application to tell if you're wearing a mask right

Follow https://imageai.readthedocs.io/en/latest/custom/ Prediction model tutorial.
Used images from https://github.com/cabani/MaskedFace-Net. Our model was trained with 2000 images, 1000 for each category. 750 train, 250 test.
Note that I used PowerRename to simplify the images' filenames.
Good = masked properly
Bad = masked improperly

## training_data_structure

```
training_data/
    json/
        model_class.json
    logs/
        */
    models/
        model_ex*.h5
    test/
        good/
            good-*.(jpg|jpeg|png)
        bad/
            bad-*.(jpg|jpeg|png)
```

## pip dependencies

```
tensorflow<2
scipy<1.5
numpy
keras
opencv-python
```
