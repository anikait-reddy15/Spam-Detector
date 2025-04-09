import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import chardet

# Detect file encoding
with open('spam.csv', 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']

print(f"File encoding: {encoding}")

#Load dataset
data = pd.read_csv('spam.csv', encoding=encoding)

# Convert labels to binary values (ham -> 0, spam -> 1)
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)

# Feature extraction using Bag of Words (CountVectorizer)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Predicting the test set results
predictions = classifier.predict(X_test_counts)

# Evaluating the classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Visualizing the percentage of spam and ham messages
labels = ['Ham', 'Spam']
values = [len(data[data['v1'] == 0]), len(data[data['v1'] == 1])]
colors = ['lightblue', 'lightcoral']
plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("Percentage of Ham and Spam Messages")
plt.show()
