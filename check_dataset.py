import pandas as pd

df = pd.read_csv("dataset/synthetic_heart_disease_dataset.csv")

print("Gender:", df["Gender"].unique())
print("Smoking:", df["Smoking"].unique())
print("Alcohol:", df["Alcohol_Intake"].unique())
print("Physical:", df["Physical_Activity"].unique())
print("Diet:", df["Diet"].unique())
print("Hypertension:", df["Hypertension"].unique())
print("Diabetes:", df["Diabetes"].unique())
print("Hyperlipidemia:", df["Hyperlipidemia"].unique())
print("Family History:", df["Family_History"].unique())
print("Previous Heart Attack:", df["Previous_Heart_Attack"].unique())