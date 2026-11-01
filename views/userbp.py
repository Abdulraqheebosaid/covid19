from flask import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

user_bp = Blueprint('user_bp', __name__)
# Structure of the Neural Network
@user_bp.route('/user')
def user():
    return render_template("user.html")


@user_bp.route('/user_home',  methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if request.form['user'] == 'user' and request.form['pwd'] == 'user':
        return render_template("predict1.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('user.html', msg=msg)

"""
@user_bp.route('/predict', methods=['POST', 'GET'])
def predict():
    pro_tweet=""
    sentiment=""
    tweet=""
    if (preprocess() == "valid"):
        #loaded_model = TheModelClass(*args, **kwargs)
        #loaded_model.load_state_dict(torch.load('model_name.pth'))
        model = torch.load("models/lstm_model.pth")
        #loaded_model=TheModelClass(*args, **kwargs)
        #loaded_model.load_state_dict(torch.load('models/lstm_model.pth'))
        tweet = request.form['text1']
        print(tweet)

        pro_tweet,sentiment=data.FinalClassifier.sentiment(model,tweet)
        return render_template('result1.html',tweet=tweet, pro_tweet=pro_tweet, sentiment=sentiment )

    else:
        return render_template('result1.html', tweet=tweet, pro_tweet=pro_tweet, sentiment=sentiment)
"""

@user_bp.route('/predict', methods=['POST', 'GET'])
def predict():
    stop_words = stopwords.words('english')
    sentiment=""
    # convert to lowercase
    text1 = request.form['text1'].lower()
    text_final = ''.join(c for c in text1 if not c.isdigit())
    # remove punctuations
    # text3 = ''.join(c for c in text2 if c not in punctuation)
    # remove stopwords
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])
    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound']) / 2, 2) * 100
    if compound>=0 and compound<=40:
        sentiment="Negative"
    elif compound>=41 and compound<=60:
        sentiment="Neutral"
    else:
        sentiment="Positive"

    return render_template('predict1.html', final=compound, text1=text_final, sentiment=sentiment)

@user_bp.route('/userlogout')
def userlogout():
    return render_template("home.html")