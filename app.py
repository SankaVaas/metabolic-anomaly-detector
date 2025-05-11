from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Your model loading and training code here (e.g., load 'model2' as in your previous code)

data = pd.read_csv('./diabetes.csv')

@app.route('/')
def index():
    return render_template('index.tsx')  # Assuming your Next.js pages are in a 'pages' folder

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        new_data = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [dpf],
            "Age": [age]
        })

        X_train = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                        "DiabetesPedigreeFunction", "Age"]][:768]
        y_train = data[["Outcome"]][:768].values.reshape(-1, 1)
        model2 = LogisticRegression(C=10.0, random_state=0, solver='liblinear')
        model2.fit(X_train, y_train)
        prediction = model2.predict(new_data)

        result = "You're suffering from diabetes" if prediction == 1 else "You're not suffering from diabetes"

        return render_template('pages/result.tsx', result=result)  # Assuming your Next.js pages are in a 'pages' folder

if __name__ == '__main__':
    app.run(debug=True)
