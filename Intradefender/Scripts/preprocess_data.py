import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# File paths
dataset_dir = "../datasets/"
train_file = os.path.join(dataset_dir, "KDDTrain+.txt")
test_file = os.path.join(dataset_dir, "KDDTest+.txt")

# Column names (based on NSL-KDD documentation)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label'
]

# Load the datasets
train_data = pd.read_csv(train_file, header=None, names=columns)
test_data = pd.read_csv(test_file, header=None, names=columns)

print("Train dataset shape:", train_data.shape)
print("Test dataset shape:", test_data.shape)

# Combine train and test data for consistent encoding
combined_data = pd.concat([train_data, test_data], axis=0)

# List of categorical features to encode
categorical_features = ['protocol_type', 'service', 'flag']

# Encode categorical features
for feature in categorical_features:
    encoder = LabelEncoder()
    combined_data[feature] = encoder.fit_transform(combined_data[feature])

# Split back into train and test data
train_data = combined_data.iloc[:train_data.shape[0], :].copy()  # Use .copy() to avoid SettingWithCopyWarning
test_data = combined_data.iloc[train_data.shape[0]:, :].copy()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Exclude the label column for scaling
numerical_features = [col for col in train_data.columns if col != 'label' and train_data[col].dtype in ['int64', 'float64']]  # Only numeric columns

# Apply scaler to both train and test sets
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

# Save the preprocessed datasets
train_data.to_csv("../datasets/preprocessed_train.csv", index=False)
test_data.to_csv("../datasets/preprocessed_test.csv", index=False)

print("Preprocessed train and test datasets saved successfully.")
