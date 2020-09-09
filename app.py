from flask import Flask,render_template,url_for,request
import pickle
import joblib
app=Flask(__name__,template_folder='templates')
app.config["DEBUG"]=True

filename="model.pkl"
model=joblib.load(open(filename,'rb'))
cv=joblib.load(open('transform.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    message=request.args['message']
    data=[message]
    vect=cv.transform(data).toarray()
    my_prediction=model.predict(vect)
    if my_prediction==1:
        result="Looking Spam, Be Safe"
    elif my_prediction==0:
        result="Not a Spam message"
    return result
    


if __name__ == "__main__":
    app.run(debug=True)
