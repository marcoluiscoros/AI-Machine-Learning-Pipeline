from django.shortcuts import render
from joblib import load
import numpy as np

# Load your saved models and mappings
models = {
    'knn': load('./savedModels/diabetes_prediction_model.joblib'),
    'decision_tree': load('./savedModels/diabetes_prediction_dt_model.joblib')
}
scalers = load('./savedModels/diabetes_prediction_model_scalers.joblib')
mappings = load('./savedModels/diabetes_prediction_model_mappings.joblib')

def predictor(request):
    return render(request, 'main.html') 

def formInfo(request):
    if request.method == 'GET':
        # Retrieve form data
        algorithm = request.GET.get('algorithm')
        gender = request.GET.get('gender')
        age = float(request.GET.get('age'))
        hypertension = int(request.GET.get('hypertension'))
        heart_disease = int(request.GET.get('heart_disease'))
        bmi = float(request.GET.get('bmi'))
        hbA1c_level = float(request.GET.get('hba1c_level'))
        glucose = float(request.GET.get('glucose'))

        # Map gender to numeric representation
        gender_numeric = mappings['gender'][gender]

        # Create a list with the input data
        datapoint = [gender_numeric, age, hypertension, heart_disease, bmi, hbA1c_level, glucose]

        # Perform scaling and prediction
        prediction = predict_diabetes(datapoint, algorithm)

        # Prepare data to be sent to the template
        context = {
            'prediction': prediction
        }

        return render(request, 'result.html', context)

def predict_diabetes(datapoint, algorithm):
    datapoint_numpy = np.array(datapoint)

    # Perform scaling for numerical features, starting from index 1
    for i, scaler in enumerate(scalers):
        datapoint_numpy[i] = scaler.transform(np.array([datapoint_numpy[i]]).reshape(-1, 1))[0, 0]

    # Select the appropriate model based on the algorithm
    model = models[algorithm]

    # Return Predicted Model
    return model.predict(datapoint_numpy.astype(np.float64).reshape(1, -1))[0]
