import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask('diabetes_app')
model = pickle.load(open('model.pkl', 'rb'))

#map url / to fuction show_predict_diabetes_form()
@app.route('/')
def show_predict_diabetes_form():
    return render_template('index.html')

#url /results is now mapped to results() function
@app.route('/result', methods=['POST'])
def results():
    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        if(prediction[0] == 0):
            return render_template('output.html',prediction_text = 'Congratulations!! You are not Diabetic!')
        else:
            return render_template('output.html', prediction_text='You are Diabetic. Please consult a Doctor.')

app.run("localhost", "3000", debug=True)