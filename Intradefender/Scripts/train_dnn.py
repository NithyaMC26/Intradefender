import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load datasets
train_data = pd.read_csv('C:/Intradefender/Datasets/preprocessed_train.csv')
test_data = pd.read_csv('C:/Intradefender/Datasets/preprocessed_test.csv')

# Check for NaN or non-numeric values and handle them
train_data = train_data.dropna().reset_index(drop=True)
test_data = test_data.dropna().reset_index(drop=True)

# Ensure all features are numeric
train_data = train_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# Extract features and labels
X_train = train_data.drop(columns=['label']).values
y_train = pd.get_dummies(train_data['label']).values
X_test = test_data.drop(columns=['label']).values
y_test = pd.get_dummies(test_data['label']).values

# Build the DNN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred))

