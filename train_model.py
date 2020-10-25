from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("training_data")
model_trainer.trainModel(num_objects=3, num_experiments=1000,
                         batch_size=32, enhance_data=True, continue_from_model="training_data/models/model_ex-067_acc-0.953125.h5")
