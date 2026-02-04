!pip install pyngrok
from flask import Flask, request
from pyngrok import ngrok
import pickle

app = Flask(__name__)

# Load ML model
with open("irrigation_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        moisture = float(request.form["moisture"])
        temp = float(request.form["temperature"])

        pred = model.predict([[moisture, temp]])[0]
        result = "Pump ON – Irrigation Required" if pred == 1 else "Pump OFF – No Irrigation"

    return f"""
    <html>
    <head><title>Smart Irrigation System</title></head>
    <body>
        <h2>Smart Irrigation System (AI)</h2>
        <form method="post">
            Soil Moisture: <input type="number" name="moisture" required><br><br>
            Temperature (°C): <input type="number" name="temperature" required><br><br>
            <input type="submit" value="Predict">
        </form>
        <h3>{result}</h3>
    </body>
    </html>
    """

# Set ngrok authtoken
ngrok.set_auth_token("39C2MtnBYHN9XZrGnAkuRoj3IGL_3mzjoZyquWkLNBxsdT9Hp")

# Open ngrok tunnel
public_url = ngrok.connect(5000)
print("🌐 Open this URL:", public_url)

app.run(port=5000)
