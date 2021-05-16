from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('App/model.pkl', 'rb'))
vectorizer = pickle.load(open('App/vectorizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    str_articles = [str(x) for x in request.form.values()]
    tra_articles = vectorizer.transform(str_articles)
    prediction = model.predict(tra_articles)
    probability = model.predict_proba(tra_articles)

    if prediction == 0:
        output = "Real News"

    else:
        output = "Fake News"

    return render_template('index.html', prediction_text='This article was predicted to be {} with a Real News Probability of {} and Fake News Probability of {}'.format(output, probability[0][0], probability[0][1]))


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
