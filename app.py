import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('tt_model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    Pregnancies = int(request.form['Pregnancies'])
    Glucose = int(request.form['Glucose'])
    BloodPressure = int(request.form['BloodPressure'])
    SkinThickness = int(request.form['SkinThickness'])
    Insulin = int(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = int(request.form['Age'])

    # Perform prediction using the input data
    prediction = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Get the output
    output = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

    # Return the prediction to the user
    return render_template('index.html', prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
