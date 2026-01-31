import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("urls.csv")


X = df[['url_length', 'special_chars', 'has_https']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

new_url = pd.DataFrame([[60, 4, 0]], columns=['url_length', 'special_chars', 'has_https'])
result = model.predict(new_url)

print("\nNew URL Analysis:")
if result[0] == 1:
    print("Result: Malicious URL")
else:
    print("Result: Benign URL")

