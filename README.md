# LSGH-CP-Fake-News-Detector
The pretrained model repository of the Filipino Fake New Detector under the study, "Fighting Disinformation in the Philippines: A Natural Language Processing Approach to Detecting Filipino Fake News".

## Setup
1. Install the required libraries listed in `requirements.txt` in a python venv.
2. Specify the Flask app using `set FLASK_APP = App/App.py`
3. run app using `flask run`

## Usage
1. Copy article from the internet (*Minimum of 100 Words*).
2. Paste the copied article in the text box.
3. Press `Predict` button.

## Replicate Results
1. Import the [Tagalog Fake News Dataset](https://github.com/jcblaisecruz02/Tagalog-fake-news) by Cruz et al.(2020) as a Pandas dataframe.
2. Insantiate the pretrained vectorizer and model using: <br>
`classifier = pickle.load(open('App/model.pkl', 'rb'))` <br>
`vectorizer = pickle.load(open('App/vectorizer.pkl', 'rb'))`
3. Vectorize the dataset using the `vectorizer.transform(df) function`
4. Perform a train_test split using the sklearn train_test_split function. <br>
`X_train, X_test, y_train, y_test = train_test_split(all_features, data.label, test_size=0.3, random_state = 88)`
5. Check Accuracy using: `print(f"Accuracy: {classifier.score(X_test, y_test):.2%}")`

### Disclaimer
<em>This work is a proof-of-concept prototype only. As such, the accuracy and correctness of the predictions of the model are not guaranteed and should not be assumed as correct. <em/>
