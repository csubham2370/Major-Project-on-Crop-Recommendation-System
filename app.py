from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# importing model
model = pickle.load(open('models/RandomForest.pkl','rb'))

# creating flask app
app = Flask(__name__, template_folder='template')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    k = request.form['Potassium']
    temperature = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N,P,k,temperature,humidity,ph,rainfall]
    features = np.array([[feature_list]]).reshape(1, -1)
    prediction = model.predict(features)[0]

    if True:

       result = "{} is a best crop to be cultivated. ".format(prediction)

    return render_template('index.html',result = result)


# python main
if __name__ == "__main__":
    app.run(debug=True)