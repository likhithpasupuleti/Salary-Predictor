import numpy as np
from flask import Flask,render_template,request,jsonify
import pickle


model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    experience = int(request.form['experience'])
    test_score= int(request.form['test_score'])
    interview_score = int(request.form['interview_score'])
        
    data = np.array([experience,test_score,interview_score])
    data1=data.reshape(1,-1)
    
    prediction = model.predict(data1)
    
    salary=round(prediction[0])
    
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(salary))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    salary = prediction[0]
    return jsonify(salary)

if __name__ == "__main__":
    app.run(debug=True)