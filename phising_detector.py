Full Python code for AI powered Phising detector:
                pip install scikit-learn pandas numpy

                #python code:
                import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Sample dataset (small, extend later)
data = {
    "url": [
        "http://192.168.0.1/login.php",
        "http://paypal.account.verify-user.com/login",
        "http://example.com",
        "https://github.com/login",
        "https://secure.bank-update.com",
        "https://www.google.com",
        "http://signin.verify.account.fakebank.com",
        "https://stackoverflow.com/questions",
        "http://login.account-checking.com",
        "https://openai.com/blog"
    ],
    "label": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1 = Phishing, 0 = Safe
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["url"], df["label"], test_size=0.3, random_state=42)

# Create pipeline: vectorizer + logistic regression
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", LogisticRegression())
])

# Train the model
model.fit(X_train, y_train)

# === Predict on a new URL ===
def check_url_ai(url):
    prediction = model.predict([url])[0]
    return "Phishing" if prediction == 1 else "Safe"

# Test it
user_url = input("Enter a URL to check with AI: ")
print(f"\nAI Model Prediction: {check_url_ai(user_url)}")
