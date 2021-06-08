from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('forest_fires.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("forestfires.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    return render_template('forestfires.html',pred='Area Burned in hectres is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)