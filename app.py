from flask import *
from pickle import load
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

model = load(open("model.pkl", "rb"))
vectorizer = load(open("vectorizer.pkl", "rb"))

sw = set(stopwords.words("english"))
ss = SnowballStemmer("english")
def clean_text(txt):
	txt = txt.lower()
	txt = word_tokenize(txt)
	txt = [t for t in txt if t not in punctuation]
	txt = [t for t in txt if t not in sw]
	txt = [ss.stem(t) for t in txt]
	txt = " ".join(txt)
	return txt

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def home():
	result = ""
	if request.method == "POST":
		news = request.form["news"]
	
		cleaned = clean_text(news)
		vec = vectorizer.transform([cleaned])
		pred = model.predict(vec)

		if pred[0] == 1:
			result = "Real News ✅"
		else:
			result = "Fake News ❌"
	return render_template("index.html", result=result)

#app.run(debug=True, use_reloader=True, port=9000)