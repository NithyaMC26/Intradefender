import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv('C:/Intradefender/Datasets/preprocessed_train.csv')
test_data = pd.read_csv('C:/Intradefender/Datasets/preprocessed_test.csv')

X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

label_encoder = LabelEncoder()
X_train['protocol_encoded'] = label_encoder.fit_transform(X_train['protocol_type'])
X_test['protocol_encoded'] = label_encoder.transform(X_test['protocol_type'])
X_train.drop('protocol_type', axis=1, inplace=True)
X_test.drop('protocol_type', axis=1, inplace=True)

non_numeric_cols = X_train.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)

for col in non_numeric_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
