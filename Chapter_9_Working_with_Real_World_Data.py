
from sklearn.linear_model import LinearRegression

# Example of housing price prediction with regression
housing_model = LinearRegression()
housing_model.fit(X_train, y_train)
housing_predictions = housing_model.predict(X_test)

# Evaluate the model
print(f"R^2: {housing_model.score(X_test, y_test)}")
