from flask import Flask, jsonify, request, render_template
import pickle
import lightgbm as lgb

model = pickle.load(open('model/trained_model.pkl','rb'))

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
    if(request.method == 'GET'):

        return render_template("predict.html")

@app.route('/predict', methods = ['POST'])
def predict():
    

    nitrogenContent = request.form.get("nitrogenContent")
    phosphorousContent = request.form.get("phosphorousContent")
    potassiumContent = request.form.get("potassiumContent")
    a_temperature = request.form.get("a_temperature")
    humidity = request.form.get("humidity")
    ph = request.form.get("ph")
    rainfall = request.form.get("rainfall")
    
    if(nitrogenContent == '' or phosphorousContent == '' or potassiumContent == '' or a_temperature == '' or humidity == '' or ph == '' or rainfall == ''):
        return jsonify({"res": 0})

    prediction = model.predict([
        [   float(nitrogenContent),
            float(phosphorousContent),
            float(potassiumContent),
            float(a_temperature),
            float(humidity),
            float(ph),
            float(rainfall)
        ]
    ])
    return jsonify({"res": prediction[0].lower()})

if __name__ == '__main__':
    app.run(debug = True)


