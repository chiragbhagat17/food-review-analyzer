from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("../models/sentiment_svm_clean.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer_clean.pkl")

@app.route('/')
def home() :
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict() :
    review = request.form.get('review')
    
    if not review or not review.strip():
        result = "Please enter a review"
        
        return render_template(
            "index.html",
            result = result
        )
    
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    decision_score = model.decision_function(review_tfidf)[0]
    
    if prediction == 1:
        result = "Positive"
        result_class = "positive"
    else : 
        result = "Negative"
        result_class = "negative"
    
    return render_template(
        'index.html',
        result=result,
        result_class=result_class,
        decision_score=decision_score
    )

if __name__ == "__main__" :
    app.run(debug=True)
