import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('taxi.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(feature) for feature in request.form.values()]
    final_featurs = [np.array(int_features)]
    prediction = model.predict(final_featurs)
    output = round(prediction[0],2)
    return render_template('index.html', prediction_text=f"Number of weekly rides should be {int(output)}")


if __name__ == '__main__':
    app.run(debug=True)


python minino
2 func
retrieve + 