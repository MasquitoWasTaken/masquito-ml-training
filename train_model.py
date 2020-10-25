from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("training_data")
model_trainer.trainModel(num_objects=3, num_experiments=100,
                         batch_size=32, enhance_data=True, continue_from_model="training_data/models/model_ex-018_acc-0.859375.h5", transfer_with_full_training=True)
