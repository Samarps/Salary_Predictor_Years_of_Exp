# Loading trained model
import joblib
m = joblib.load("trainedmodel.pk1")
print("Trained model loaded...\n")

# Prediction
p = int(input("Enter no. of years of experience: "))
s = m.predict([[p]])
predicted_salary = round(s[0],2)
print("Predicted Salary = ", predicted_salary, " INR")