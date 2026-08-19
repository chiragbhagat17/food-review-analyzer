# 🍔 Food Review Analyzer

A machine learning web application that analyzes food/product reviews and predicts whether the review is **Positive** or **Negative**.

The project uses **TF-IDF text features** and a **Linear SVM classifier**, with Flask providing the web interface and backend.

## 🚀 Features

- Sentiment classification of food reviews
- TF-IDF based text representation
- Unigram, bigram and trigram experiments
- Comparison of multiple ML models
- Hyperparameter tuning for Linear SVM
- Class imbalance handling using `class_weight="balanced"`
- Error analysis using SVM decision scores
- Duplicate review analysis
- Negation handling experiment
- Flask web application
- Displays prediction and SVM decision score

## 🧠 Machine Learning Approach

The main pipeline is:

```text
Review
   ↓
Text preprocessing
   ↓
TF-IDF Vectorization
   ↓
Linear SVM
   ↓
Positive / Negative

Models Compared
Model	Accuracy
Logistic Regression	91.47%
Linear SVM	94.27%
Naive Bayes	84.29%

Linear SVM performed the best.

N-gram Experiments
Features	Accuracy
Unigram	90.64%
Bigram	94.35%
Trigram	94.29%

Bigram features performed best.

SVM Hyperparameter Tuning

Different values of C were tested using 5-fold cross-validation.

The best value was:

C = 0.5

Best cross-validation macro F1:

0.8926
Final Model

The final model uses:

TF-IDF
ngram_range = (1, 2)


Linear SVM
C = 0.5
class_weight = "balanced"

Final test performance was approximately:

Accuracy: 94.36%
Macro F1: 0.89
🌐 Web Application

The Flask application allows a user to enter a review and receive a sentiment prediction.

Example:

User Review
     ↓
Flask
     ↓
TF-IDF Vectorizer
     ↓
Linear SVM
     ↓
Positive / Negative

The application also displays the model's decision score.

A score above 0 is classified as positive, while a score below 0 is classified as negative.

📁 Project Structure
food-review-analyzer/
│
├── app/
│   ├── app.py
│   ├── static/
│   │   └── styles.css
│   └── templates/
│       └── index.html
│
├── data/
│   └── reviews.csv
│
├── models/
│   ├── sentiment_svm_clean.pkl
│   └── tfidf_vectorizer_clean.pkl
│
├── notebooks/
│   └── 01_sentiment_model.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md

The original dataset is not included in the GitHub repository because of its large file size.

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
Flask
HTML
CSS
Jupyter Notebook
▶️ How to Run
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/food-review-analyzer.git
cd food-review-analyzer
2. Create a virtual environment
python -m venv venv
3. Activate it

Windows PowerShell:

venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
5. Run Flask
cd app
python app.py
6. Open the application
http://127.0.0.1:5000
📊 Evaluation

The final model achieved approximately:

94.36% accuracy
0.89 macro F1-score
Strong performance on both positive and negative classes despite class imbalance.
🔍 Future Improvements
Experiment with more advanced NLP preprocessing
Try word embeddings
Experiment with transformer-based models
Improve handling of complex negation
Add probability calibration
Deploy the application online
Add a REST API using FastAPI
👨‍💻 Author

Chirag

Food Review Analyzer — Sentiment Analysis Project