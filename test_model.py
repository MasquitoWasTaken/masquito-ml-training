from imageai.Prediction.Custom import CustomImagePrediction
import os

test_image = "test-2.jpg"
model = "model_ex-005_acc-0.945312.h5"
execution_path = os.getcwd()

# Set up predictor class
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(
    execution_path, "training_data/models/" + model))
prediction.setJsonPath(os.path.join(
    execution_path, "training_data/json/model_class.json"))
prediction.loadModel(num_objects=3)

# Get and display results
predictions, probabilities = prediction.predictImage(
    os.path.join(execution_path, "test_images/" + test_image), result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
