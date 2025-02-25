import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
train_data = pd.read_csv('C:/Intradefender/Datasets/preprocessed_train.csv')
test_data = pd.read_csv('C:/Intradefender/Datasets/preprocessed_test.csv')

X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred))
