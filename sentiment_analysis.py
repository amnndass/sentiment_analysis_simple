# from sklearn.linear_model import LogisticRegression
# import pandas as pd

# # Just 4 tweets: 2 happy, 2 unhappy
# data = {
#     'text': [
#         "I love this phone, it is amazing",
#         "Best battery life ever",
#         "Worst customer service ever",
#         "This phone is terrible and broken"
#     ],
#     'label': [1, 1, 0, 0]   # 1 = Positive, 0 = Negative
# }

# df = pd.DataFrame(data)
# print(df)

# X_text = df['text']
# y = df['label']

# print(X_text)
# print(y)

# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(stop_words="english")

# X = vectorizer.fit_transform(X_text)

# print("the words the computer found are")
# print(vectorizer.get_feature_names_out())
# print(X.toarray())

# from sklearn.linear_model import LinearRegression

# model = LogisticRegression()
# model.fit(X,y)

# new_tweet = ['i love the battery']

# new_x = vectorizer.transform(new_tweet)

# prediction = model.predict_proba(new_x)
# print("the prediction is ", prediction)

# bad_tweet = ['worst service, it is broken']
# bad_x = vectorizer.transform(bad_tweet)

# print(bad_tweet[0])

# prediction = model.predict_proba(bad_x)
# print("the prediction is ", prediction)

# words = vectorizer.get_feature_names_out()
# weights = model.coef_[0]

# for word, weight in zip(words, weights):
#     print(word, " : ", f"{weight:.3f}")

#bigger dataset
import pandas as pd


df = pd.read_csv("tweets.csv")
print('dataset shape', df.shape)
print(df.head())

from sklearn.model_selection import train_test_split

#seperate
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

print(len(X_train), ' : training data')
print(len(X_test), ' : testing data')

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

print(X_train_vec.shape)
print(X_test_vec.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(C=5.0)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

#naive bayes
# from sklearn.naive_bayes import MultinomialNB

# model = MultinomialNB()
# model.fit(X_train_vec, y_train)

# y_pred = model.predict(X_test_vec)

# from sklearn.metrics import accuracy_score

print(vectorizer.get_feature_names_out())

print(list(y_test))
print(y_pred)

print(accuracy_score(y_test, y_pred))

exam_review = pd.DataFrame({
    'tweet': X_test.values,
    'Real': y_test.values,
    'predicted': y_pred
})

print(exam_review)