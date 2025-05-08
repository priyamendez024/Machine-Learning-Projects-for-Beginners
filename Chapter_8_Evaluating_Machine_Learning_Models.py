
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix:
", cm)

print("Classification Report:
", classification_report(y_test, y_pred_dt))
