import pandas as pd

features_df = pd.read_csv('UNSW-NB15 Dataset/UNSW-NB15_features.csv', encoding='ISO-8859-1')
column_names = features_df['Name'].values

df1 = pd.read_csv('UNSW-NB15 Dataset/UNSW-NB15_1.csv', header=None)
df2 = pd.read_csv('UNSW-NB15 Dataset/UNSW-NB15_2.csv', header=None)
df3 = pd.read_csv('UNSW-NB15 Dataset/UNSW-NB15_3.csv', header=None)
df4 = pd.read_csv('UNSW-NB15 Dataset/UNSW-NB15_4.csv', header=None)

combined_dataset = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
combined_dataset.columns = column_names

combined_dataset['Stime'] = pd.to_datetime(combined_dataset['Stime'], unit='s')  # Convert start time
combined_dataset['Ltime'] = pd.to_datetime(combined_dataset['Ltime'], unit='s')  # Convert last time

combined_dataset = combined_dataset.sort_values(by='Stime').reset_index(drop=True)

# combined_dataset.to_csv('raw_data.csv')   # This is the dataset including *all* features

edge_attributes = ['sbytes', 'dbytes', 'Spkts', 'Dpkts',  # Volume properties
                       'dur', 'Sintpkt', 'Dintpkt', 'tcprtt', # Time properties
                    #    'proto', 'service', 'state',           # Protocol & Service properties
                       'Sjit', 'Djit',]

node_attributes = ['srcip', 'dstip']    # Source and destination IP addresses for nodes

relevant_data = pd.concat([combined_dataset[node_attributes], combined_dataset[edge_attributes]], axis=1)
# relevant_data.to_csv('feature_reduced_dataset.csv')      # This is the dataset, with *relevant* features, before further preprocessing

# Normalizing numeric features:
numeric_features = relevant_data.select_dtypes(exclude='object') # removing categorical features

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numeric_features_normalized = scaler.fit_transform(numeric_features)
numeric_features_normalized_df = pd.DataFrame(numeric_features_normalized, columns=numeric_features.columns)

# print(relevant_data.columns)

# Attaching labels:
relevant_data['Label'] = combined_dataset['Label']

# Replace the original numeric columns in 'relevant_data' with the normalized versions
processed_data = relevant_data.copy()
processed_data.loc[:, numeric_features_normalized_df.columns] = numeric_features_normalized_df.values
# Convert boolean (True/False) to integer (0/1)
processed_data = processed_data.astype({col: 'int32' for col in processed_data.select_dtypes(include=['bool']).columns}) # Change true/false to 1/0
# print(processed_data.head(10))

# Encode srcip and dstip into unique node indices
node_mapping = {ip: idx for idx, ip in enumerate(pd.concat([processed_data['srcip'], processed_data['dstip']]).unique())}
processed_data['srcip_index'] = processed_data['srcip'].map(node_mapping)
processed_data['dstip_index'] = processed_data['dstip'].map(node_mapping)
print(node_mapping)
# unique_ips = set(processed_data['srcip_index']).union(set(processed_data['dstip_index']))
# print(unique_ips)

# Time based split - first 80% of the timeline is used for training

train_size = int(len(processed_data) * 0.8) # Calculate the split index

# Training set: Filter only benign flows
train_data = processed_data.iloc[:train_size]
train_data = train_data[combined_dataset['Label'] == 0].drop(columns=['Label']) # Label 0 = benign

# Create a validation set from train_data (using the last 20% of train_data for validation)
train_val_split_index = int(len(train_data) * 0.8)  # 80% of the benign training flows remain for training
train_data_new = train_data.iloc[:train_val_split_index]  # Training set (80% of benign flows)
val_data = train_data.iloc[train_val_split_index:]         # Validation set (20% of benign flows)

print("Training set shape:", train_data_new.shape)
print("Validation set shape:", val_data.shape)

# Testing set: Mix of benign and attack flows
test_data = processed_data.iloc[train_size:]

processed_data.to_csv('processed_data.csv')
train_data.to_csv('train_data.csv')
val_data.to_csv('val_data.csv')
test_data.to_csv('test_data.csv')
