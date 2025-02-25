import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  

train_data = pd.read_csv('C:/Intradefender/Datasets/preprocessed_train.csv')
X = train_data.drop('label', axis=1)
y = train_data['label']
print("Columns with non-numeric values:", X.select_dtypes(include=['object']).columns)
categorical_columns = X.select_dtypes(include=['object']).columns

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for col in categorical_columns:
    X[col] = encoder.fit_transform(X[col])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Model training complete.")
y_pred = rf_model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

print("Classification Report:\n", classification_report(y_val, y_pred))
joblib.dump(rf_model, 'random_forest_model.pkl')