
from sklearn.svm import SVR

# Support Vector Machine for Regression
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Model evaluation
print(f"R^2: {svm_model.score(X_test, y_test)}")
