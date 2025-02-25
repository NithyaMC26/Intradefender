from joblib import load
import pandas as pd

df = pd.read_csv('C:/Intradefender/Datasets/aligned_live.csv')
model = load('C:/Intradefender/Results/random_forest_model.pkl')
predictions = model.predict(df)
with open('C:/Intradefender/Results/predictions_log.txt', 'w') as log_file:
    for i, pred in enumerate(predictions):
        if pred == 1:  # Assuming "1" indicates malicious traffic
            log_file.write(f"Alert: Intrusion detected in packet {i}\n")
print("Predictions saved to log file.")
