# CardioVascular-Risk-Assessment
A web-based cardiovascular risk assessment system that predicts heart disease using patient health parameters and a Random Forest machine learning model.
# Cardiovascular Risk Assessment using Machine Learning

## Project Overview

This project is a machine learning-based web application that predicts the risk of heart disease using patient health information. It is developed using Python, Scikit-learn, and Streamlit. The application allows users to enter patient details and predicts whether the patient is at low or high risk of heart disease.

The prediction model is built using the Random Forest Classifier and trained on a synthetic heart disease dataset.

---

## Features

- Predicts heart disease risk
- Interactive web application using Streamlit
- Random Forest machine learning model
- Displays prediction confidence
- Shows risk score
- Provides basic health recommendations

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

---

## Project Structure

```
Cardiovascular-Risk_Assessment/
│
├── dataset/
│   └── synthetic_heart_disease_dataset.csv
│
├── app.py
├── train.py
├── heart_model.pkl
├── encoders.pkl
├── feature_names.pkl
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses a synthetic heart disease dataset containing around **50,000 records**.

### Input Features

- Age
- Gender
- Weight
- Height
- BMI
- Smoking
- Alcohol Intake
- Physical Activity
- Diet
- Stress Level
- Hypertension
- Diabetes
- Hyperlipidemia
- Family History
- Previous Heart Attack
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Heart Rate
- Blood Sugar Fasting
- Total Cholesterol

### Target Variable

**Heart_Disease**

- 0 – Low Risk
- 1 – High Risk

---

## Machine Learning Model

The project uses the **Random Forest Classifier** for prediction.

The training process includes:

1. Loading the dataset
2. Handling missing values
3. Encoding categorical data
4. Splitting the dataset into training and testing sets
5. Training the Random Forest model
6. Evaluating the model
7. Saving the trained model

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Cardiovascular-Risk_Assessment.git
```

Go to the project folder:

```bash
cd Cardiovascular-Risk_Assessment
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the Model

```bash
python train.py
```

This creates:

- heart_model.pkl
- encoders.pkl
- feature_names.pkl

### Run the Application

```bash
streamlit run app.py
```

After running the command, open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## Output

The application provides:

- Heart disease prediction
- Prediction confidence
- Risk score
- Health recommendations

---

## Future Improvements

- Improve model accuracy using different algorithms
- Compare multiple machine learning models
- Deploy the application online
- Add patient history management
- Integrate real healthcare datasets

---

## Author

**Roshni J**

---

## License

This project is created for academic and learning purposes.
