from flask import Flask, request, jsonify
import pickle
import numpy as np
from pathlib import Path

app = Flask(__name__)

model_path = Path(__file__).parent / "model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict():
    data = request.json["features"]
    prediction = model.predict(np.array([data]))

    return jsonify({
        "prediction": prediction.tolist()
    })

if __name__ == "__main__":
    app.run(debug = True)