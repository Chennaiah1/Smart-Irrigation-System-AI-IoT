import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load data
data = pd.read_csv("../data/irrigation_data.csv")

# Plot moisture distribution
plt.figure()
sns.histplot(data["soil_moisture"], kde=True)
plt.title("Original Moisture Distribution")
plt.xlabel("Soil Moisture")
plt.ylabel("Frequency")
plt.show()

# Plot temperature distribution
plt.figure()
sns.histplot(data["temperature"], kde=True)
plt.title("Original Temperature Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.show()

# Prepare data
X = data[["soil_moisture", "temperature"]]
y = data["pump_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
with open("../model/irrigation_model.pkl", "rb") as file:
    model = pickle.load(file)

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




