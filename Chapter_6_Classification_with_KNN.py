
from sklearn.neighbors import KNeighborsClassifier

# KNN Model Training
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Accuracy Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn) * 100:.2f}%")
