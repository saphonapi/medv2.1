import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
pmodel = pickle.load(open('pmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = pmodel.predict(final_features)
    output = round(prediction[0])
    
    if output == 0:output = "Death"        
    if output == 1:output = "Not improve"        
    if output == 2:output = "Improve"       

    return render_template('home.html', prediction_text='Your status is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = pmodel.predict([np.array(list(data.values()))])
    output = prediction[0]    
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)