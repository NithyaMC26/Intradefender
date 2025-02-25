import pyshark
import pandas as pd

# Path to the PCAP file
pcap_file = 'C:/Intradefender/Datasets/samdata.pcap'
# Load the PCAP file using pyshark
cap = pyshark.FileCapture(pcap_file)
# Extract relevant fields from packets
data = []
for packet in cap:
    try:
        data.append({
            'src_ip': packet.ip.src,
            'dst_ip': packet.ip.dst,
            'protocol': packet.transport_layer,
            'length': int(packet.length),
        })
    except AttributeError:
        # Skip packets without the required attributes
        continue

# Convert to a DataFrame
df = pd.DataFrame(data)
# Save to a CSV file for further processing
df.to_csv('C:/Intradefender/Datasets/preprocessed_live.csv', index=False)
print("Parsed live traffic saved to 'preprocessed_live.csv'")
