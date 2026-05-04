A simple machine learning web application that predicts whether a movie review is Positive 😊 or Negative 😡 using Natural Language Processing and Logistic Regression.

The model is deployed using Streamlit Cloud for real-time predictions.


🚀 Live Demo

👉 https://your-streamlit-app-link.streamlit.app

📌 Project Overview

This project performs sentiment analysis on movie reviews using the IMDB dataset.
It converts text data into numerical features using TF-IDF Vectorization and trains a Logistic Regression model to classify sentiment.

The trained model is integrated into a Streamlit web app for interactive predictions.



🧠 Machine Learning Workflow
Load IMDB dataset
Text preprocessing (stopwords removal, vectorization)
Feature extraction using TF-IDF
Train-test split (80/20)
Train Logistic Regression model
Evaluate accuracy (~88%)
Save model using joblib
Deploy using Streamlit

🛠️ Tech Stack
Python 🐍
Pandas
NumPy
Scikit-learn
Joblib
Streamlit

📁 Project Structure
sentiment_analysis/
│── app.py                  # Streamlit web app
│── train_model.py          # Model training script
│── sentiment_pipeline.pkl  # Trained ML model
│── requirements.txt        # Dependencies
│── IMDB Dataset.csv        # Dataset (optional)

📊 Model Performance
Algorithm: Logistic Regression
Feature Extraction: TF-IDF
Accuracy: ~88%


🎯 Features
Input movie review text
Predict sentiment (Positive / Negative)
Show confidence score
Clean and modern UI (Netflix-style theme)
Fast real-time predictions

🎯 Features
Input movie review text
Predict sentiment (Positive / Negative)
Show confidence score
Clean and modern UI (Netflix-style theme)
Fast real-time predictions

joblib.dump(model, "sentiment_pipeline.pkl")

It includes:

TF-IDF Vectorizer
Logistic Regression Classifier

📌 Future Improvements
Add Deep Learning (LSTM / BERT)
Improve accuracy with hyperparameter tuning
Add word cloud visualization
Support multi-language reviews

👨‍💻 Author

Girish Kunar.T
student.
