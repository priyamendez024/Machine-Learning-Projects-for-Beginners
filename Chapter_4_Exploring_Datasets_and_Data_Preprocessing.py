
# Handling missing values, encoding categorical variables, feature scaling
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Example of encoding categorical data (Gender column)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
