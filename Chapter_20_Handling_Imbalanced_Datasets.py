
from imblearn.over_sampling import SMOTE

# Applying SMOTE for handling imbalanced dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
