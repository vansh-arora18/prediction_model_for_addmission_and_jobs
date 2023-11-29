import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

FILE1 = 'model.pkl'
FILE2 = 'model2.pkl'
FILE3 = 'model3.pkl'


app = Flask(__name__)
model = pickle.load(open('/Users/vansharora/Downloads/Student_Admission_And_Jobs_Prediction-gh-pages/pycharm/model.pkl','rb'))
model2 = pickle.load(open('/Users/vansharora/Downloads/Student_Admission_And_Jobs_Prediction-gh-pages/pycharm/model2.pkl', 'rb'))
model3 = pickle.load(open('/Users/vansharora/Downloads/Student_Admission_And_Jobs_Prediction-gh-pages/pycharm/model3.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/Admissions.html")
def admissions():
    return render_template('Admissions.html')


@app.route("/Jobs.html")
def jobs():
    return render_template('Jobs.html')


@app.route("/AboutUs.html")
def About():
    return render_template('AboutUs.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = [request.form.values()]

    colleges = {1: 'Ahemedabad IT', 2: 'BIT Mesra', 3: 'BITS pilani', 4: 'BMS college of ENGG', 5: 'DTU delhi',
                6: 'HBUT kanpur', 7: 'IIEST shibpur', 8: 'IIIT hydrabad', 9: 'IIT Bombay', 10: 'IIT bhilai',
                11: 'IIT delhi', 12: 'IIT goa', 13: 'IIT guwahati', 14: 'IIT hydrabad', 15: 'IIT indore',
                16: 'IIT jammu', 17: 'IIT jodhpur', 18: 'IIT kanpur', 19: 'IIT kharagpur', 20: 'IIT mandi',
                21: 'IIT palakkad', 22: 'IIT ropar', 23: 'IIT tirupati', 24: 'Jadavpur Univversity',
                25: 'KLEF hydrabad',
                26: 'MNIT jaipur', 27: 'MNNIT allahabad', 28: 'MSIT', 29: 'Manipal IT', 30: 'NIT trichy',
                31: 'NIT warangal', 32: 'NMIMS', 33: 'Netaji Subhas IT', 34: 'S O A  university', 35: 'SRMIST chennai',
                36: 'SSN college of ENGG', 37: 'University college of ENGG', 38: 'VIT vellore'}

    features_name = ['AIEEE Rank']
    df = pd.DataFrame(input_data, columns=features_name)
    college = model.predict(df)
    return render_template('index.html',
                           prediction_text='You may have a chance to get in {}'.format(colleges[int(college)]))


@app.route('/predictadmission', methods=['POST'])
def predictadmission():
    input_data = [request.form.values()]

    features_name = ['COURSE']
    df = pd.DataFrame(input_data, columns=features_name)
    college = model2.predict(df)

    return render_template('Admissions.html',
                           prediction_text="maximum number of admissions can be {}".format(int(college)))


@app.route('/predictjob', methods=['POST'])
def predictjob():
    input_data = [request.form.values()]

    features_name = ['COURSE']
    df = pd.DataFrame(input_data, columns=features_name)
    college = model3.predict(df)

    return render_template('Jobs.html',
                           prediction_text="Expected average salary p.a in this course can be Rs.{}".format(int(college)))




if __name__ == "__main__":
    app.run(debug=True)
