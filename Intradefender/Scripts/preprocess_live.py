import pandas as pd

df = pd.read_csv('C:/Intradefender/Datasets/preprocessed_live.csv')
trained_features = pd.read_csv('C:/Intradefender/Datasets/preprocessed_train.csv').columns[:-1]  # Exclude 'label'

for feature in trained_features:
    if feature not in df.columns:
        df[feature] = 0  

df = df[trained_features]
df.to_csv('C:/Intradefender/Datasets/aligned_live.csv', index=False)
print("Live data aligned with training features.")

train_features = pd.read_csv('C:/Intradefender/Datasets/preprocessed_train.csv').columns[:-1]  # Exclude 'label'
live_features = pd.read_csv('C:/Intradefender/Datasets/aligned_live.csv').columns

print("Missing in live data:", set(train_features) - set(live_features))
print("Extra in live data:", set(live_features) - set(train_features))

