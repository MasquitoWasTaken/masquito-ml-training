from imageai.Prediction.Custom import CustomImagePrediction
import os
import time

model = "model_ex-067_acc-0.953125.h5"
execution_path = os.getcwd()

# Set up predictor class
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(
    execution_path, "training_data/models/" + model))
prediction.setJsonPath(os.path.join(
    execution_path, "training_data/json/model_class.json"))
prediction.loadModel(num_objects=3)

while True:
    filename = input("Test file: ")

    # Get and display results, time the process
    t_start = int(round(time.time() * 1000))
    predictions, probabilities = prediction.predictImage(
        os.path.join(execution_path, "test_images/" + filename), result_count=5)
    t_stop = int(round(time.time() * 1000))
    t_delta = t_stop - t_start

    print("Running time:", str(t_delta) + "ms")
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction + ":", eachProbability)
