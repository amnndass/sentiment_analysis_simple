from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("reviews_1000.csv")

print(df.head())
print(len(df))
print(df.isnull().sum())

print(df['label'].value_counts())

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

print(len(X_train), len(X_test))
print(y_test.value_counts(normalize=True))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

custom_stop_words = list(text.ENGLISH_STOP_WORDS)

vital_words = {'not', 'no', 'never', 'neither', 'nor', 'but', 'however', 'very', 'can'}
custom_stop_words = [w for w in custom_stop_words if w not in vital_words]

vectorizer = TfidfVectorizer(stop_words=custom_stop_words , min_df=2, ngram_range=(1,2), sublinear_tf=True)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(X_train_vec.shape)

print(vectorizer.get_feature_names_out())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

model = LogisticRegression(C=2.0)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

errors = pd.DataFrame({
    'review': X_test,
    'Actual': y_test,
    'predicted': y_pred
})

misclassified = errors[errors['Actual'] != errors['predicted']]

print(len(misclassified))

for idx, row in misclassified.head(10).iterrows():
    print(f"actual: {row['Actual']} | predicted: {row["predicted"]}")
    print(f"reivew: {row['review']}")


#comparing three models
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

candidates = {
    'Logistic Regression': LogisticRegression(C=2.0),
    'Multinominal Naive Bayes': MultinomialNB(alpha=0.5),
    'Linear SVM': LinearSVC(C=1.0, random_state=42)
}

print("benchmarking models")

for name, clf in candidates.items():
    clf.fit(X_train_vec, y_train)
    score = clf.score(X_test_vec, y_test)
    print(f"{name} Accuracy: {score *100:.2f}")


def analyze_review(text):
    vec = vectorizer.transform([text])

    probabilites = model.predict_proba(vec)[0]
    neg_conf = probabilites[0] * 100
    pos_conf = probabilites[1] * 100

    prediction = "positive" if pos_conf >= 50 else "negative"
    confidence = max(pos_conf, neg_conf)

    print(f"review {text}")
    print(f"prediction {prediction} ({confidence:.1f}%) confidence")
    print(f"details: positive:{pos_conf:.1f}% | negative{neg_conf:.1f}%")
    print("\n")

analyze_review("good phone would buy again, but good battery life")
analyze_review("The battery life is decent, but the screen cracked on day one.")
analyze_review("Hands down the most delicious pasta I have tasted in years.")
analyze_review("It was not bad at all, actually quite enjoyable.")

import joblib
from sklearn.pipeline import Pipeline

production_pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('classifier', LogisticRegression(C=2.0))
])

production_pipeline.fit(X_train, y_train)

model_filename = 'sentiment_pipeline.joblib'
joblib.dump(production_pipeline, model_filename)

print("production pipeline successfully exported to : {model_filename}")