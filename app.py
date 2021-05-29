# template modified from https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4
from flask import Flask, request, jsonify, render_template
from flask_recaptcha import ReCaptcha
import pickle
import numpy as np
import os

app = Flask(__name__)
recaptcha = ReCaptcha(app=app)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app.config.update(dict(
    RECAPTCHA_ENABLED=True,
    RECAPTCHA_SITE_KEY=os.environ.get('RECAPTCHA_SITE_KEY'),
    RECAPTCHA_SECRET_KEY=os.environ.get('RECAPTCHA_SECRET_KEY'),
))

recaptcha = ReCaptcha()
recaptcha.init_app(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if recaptcha.verify():
        str_articles = [request.form['article']]
        tra_articles = vectorizer.transform(str_articles)
        prediction = model.predict(tra_articles)
        probability = model.predict_proba(tra_articles)
        if not prediction:
            output = "Real News"
        else:
            output = "Fake News"

        return render_template('index.html', prediction_text='This article was predicted to be probably {} based on '
                                                             'the word choice and writing style.'.format(output,
                                                                                                         probability[0][
                                                                                                             0],
                                                                                                         probability[0][
                                                                                                             1]))
    else:
        return render_template('index.html', prediction_text='Recaptcha Failed.')


@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run()
