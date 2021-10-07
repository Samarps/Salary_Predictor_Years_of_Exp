import pandas

# Loading the Dataset
db = pandas.read_csv("SalaryData.csv")
print("Salary Dataset loaded...")

x = db[["YearsExperience"]]
y = db["Salary"]

# To Train Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
print("Model trained...")

# Saving the model
import joblib
joblib.dump(model, "trainedmodel.pkl")
print("Model saved...")
