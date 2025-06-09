from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("stroke_model.pkl")

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = {
            "gender": request.form["gender"],
            "age": float(request.form["age"]),
            "hypertension": int(request.form["hypertension"]),
            "heart_disease": int(request.form["heart_disease"]),
            "ever_married": request.form["ever_married"],
            "work_type": request.form["work_type"],
            "Residence_type": request.form["residence_type"],
            "avg_glucose_level": float(request.form["avg_glucose_level"]),
            "bmi": float(request.form["bmi"]),
            "smoking_status": request.form["smoking_status"]
        }

        input_df = pd.DataFrame([data])

        # Perform encoding exactly as done during training
        # For simplicity, assume you're using pd.get_dummies with same columns
        input_df = pd.get_dummies(input_df)

        # Align input to training columns (adjust "model_columns.pkl" if needed)
        model_columns = joblib.load("model_columns.pkl")
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        result = "Yes" if prediction == 1 else "No"
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
