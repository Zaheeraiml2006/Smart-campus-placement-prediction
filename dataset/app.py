from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model, scaler and encoders
model = pickle.load(open("placement_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    student_id = float(request.form["Student_ID"])
    age = float(request.form["Age"])

    gender = encoders["Gender"].transform(
        [request.form["Gender"]]
    )[0]

    degree = encoders["Degree"].transform(
        [request.form["Degree"]]
    )[0]

    branch = encoders["Branch"].transform(
        [request.form["Branch"]]
    )[0]

    cgpa = float(request.form["CGPA"])
    internships = float(request.form["Internships"])
    projects = float(request.form["Projects"])
    coding = float(request.form["Coding_Skills"])
    communication = float(request.form["Communication_Skills"])
    aptitude = float(request.form["Aptitude_Test_Score"])
    soft_skills = float(request.form["Soft_Skills_Rating"])
    certifications = float(request.form["Certifications"])
    backlogs = float(request.form["Backlogs"])

    # Create input data in exact dataset order
    data = [[
        student_id,
        age,
        gender,
        degree,
        branch,
        cgpa,
        internships,
        projects,
        coding,
        communication,
        aptitude,
        soft_skills,
        certifications,
        backlogs
    ]]

    # Convert to DataFrame
    data = pd.DataFrame(data, columns=[
        "Student_ID",
        "Age",
        "Gender",
        "Degree",
        "Branch",
        "CGPA",
        "Internships",
        "Projects",
        "Coding_Skills",
        "Communication_Skills",
        "Aptitude_Test_Score",
        "Soft_Skills_Rating",
        "Certifications",
        "Backlogs"
    ])

    # Scale input
    data = scaler.transform(data)

    # Prediction
    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Placed ✅"
    else:
        result = "Not Placed ❌"

    return render_template(
        "index.html",
        prediction_text=result
    )


if __name__ == "__main__":
    app.run(debug=True)