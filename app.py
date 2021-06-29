import numpy as np
from flask import Flask,request,jsonify,render_template
from recommend_movie import recommend_movie
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
recommend_model=pickle.load(open('movie_recommend.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_featurs=[int(x) for x in request.form.values()]
    final_featurs=[np.array(int_featurs)]
    prediction=model.predict(final_featurs)

    output=round(prediction[0],2)

    return render_template('prediction.html', prediction_text='Prediction of model $ {}'.format(output))
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
@app.route('/recommend',methods=['POST'])
def recommend():
    movieName=request.form["moive"]
    moive=recommend_movie(movieName)   
    return render_template('recommend.html',moive=moive)
@app.route('/recommend_api',methods=['POST'])
def recommend_api():
    data=request.get_json(force=True)
    movieName=np.array(data)
    moive=recommend_movie(movieName) 
    return jsonify(moive)

if __name__ == "__main__":
    app.run()
