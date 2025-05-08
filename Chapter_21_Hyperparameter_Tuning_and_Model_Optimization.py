
from sklearn.model_selection import RandomizedSearchCV

# RandomizedSearchCV for tuning RandomForest model
param_dist = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]}
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_dist, n_iter=100)
random_search.fit(X_train, y_train)
