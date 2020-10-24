from imageai.Prediction.Custom import CustomImagePrediction
import os

test_image = "test-1.jpg"
model = "model_ex-023_acc-0.993750.h5"
execution_path = os.getcwd()

# Set up predictor class
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(
    execution_path, "training_data/models/" + model))
prediction.setJsonPath(os.path.join(
    execution_path, "training_data/json/model_class.json"))
prediction.loadModel(num_objects=2)

# Get and display results
predictions, probabilities = prediction.predictImage(
    os.path.join(execution_path, "test_images/" + test_image), result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
