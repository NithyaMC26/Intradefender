import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv('C:/Intradefender/Datasets/preprocessed_train.csv')
test_data = pd.read_csv('C:/Intradefender/Datasets/preprocessed_test.csv')

X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']
print("Class Distribution in Training Data:\n", y_train.value_counts())

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



knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Evaluate
y_pred = knn_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
