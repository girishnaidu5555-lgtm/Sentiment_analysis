import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

X = df["review"]
y = df["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=3000
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

# TRAIN the model (VERY IMPORTANT)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
with open("sentiment_pipeline.pkl", "wb") as f:
    joblib.dump(model, f)
    
print("Model saved successfully!")
