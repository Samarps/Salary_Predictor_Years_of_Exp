import pandas

# Load Dataset
db = pandas.read_csv("SalaryData.csv")
print("Salary Dataset loaded...")

x = db[["YearsExperience"]]
y = db["Salary"]

# Train Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
print("Model trained...")

# Saving model
import joblib
joblib.dump(model, "trainedmodel.pkl")
print("Model saved...")
