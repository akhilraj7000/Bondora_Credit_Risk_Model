import numpy as np
from flask import Flask, request, render_template
import pickle
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,30)
    loaded_model = pickle.load(open("ensemble.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

def ValuePredictor2(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,30)
    loaded_model = pickle.load(open("linear.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
@app.route('/predict', methods = ['POST'])
def result():
    if request.method == 'POST':

        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        print(to_predict_list)
        result1 = ValuePredictor(to_predict_list)
        result = ValuePredictor2(to_predict_list)
        features = [np.array(result)]
        var1 = int(features[0][0])
        var2 = int(features[0][1])
        var3 = (round(features[0][2], 5))

        print(var1)
        print(var2)
        print(var3)
        if int(result1)== 0:
            prediction ='Applicant may not default'
            print(prediction)
        else:
            prediction =' Applicant may default!!!'
            print(prediction)

    return render_template("index.html",prediction_text='The loan {}'.format(prediction),
                           prediction_EMI='Preferred Monthly Payment: {}'.format(var1),
                           prediction_ROI='Preferred return on Investment: {} %'.format(var2),
                           prediction_ELA='Eligible Loan Amount is: {}'.format(var3))
    print(result)

if __name__ == "__main__":
    app.run(debug=True)