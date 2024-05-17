import os
from flask import Flask, render_template, request
import pandas as pd
import pickle
import sklearn
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))
    print(locations, bhk, bath, sqft)
    input1 = pd.DataFrame({'location': [locations],
                           'total_sqft': [sqft],
                           'bath': [bath],
                           'BHK': [bhk]})
    prediction = pipe.predict(input1)[0]
    prediction = prediction * 100000
    return str(np.round(prediction, 2))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)





