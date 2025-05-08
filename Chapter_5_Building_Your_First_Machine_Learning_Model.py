
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Model Training and Prediction
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
