
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
