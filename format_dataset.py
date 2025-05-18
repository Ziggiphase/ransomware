import pandas as pd
import numpy as np

# Load the original dataset
df = pd.read_csv('attached_assets/entropy_dataset.csv')

# Create dummy variables for file types (extract from filename column)
df['file_type_txt'] = df['filename'].str.contains('.txt').astype(int)
df['file_type_doc'] = df['filename'].str.contains('.docx').astype(int)
df['file_type_mp4'] = df['filename'].str.contains('.mp4').astype(int)
df['file_type_pdf'] = df['filename'].str.contains('.pdf').astype(int)

# Encode family names to numbers (1-18)
family_mapping = {name: i+1 for i, name in enumerate(df['family'].unique())}
df['family_label'] = df['family'].apply(lambda x: family_mapping[x])

# Create multiple entropy features by adding small random variations to the original entropy
# This simulates having multiple entropy-based features
np.random.seed(42)  # For reproducibility
num_features = 10

# Create entropy features with controlled variations
for i in range(num_features):
    # Different variations for different feature types
    if i < 3:
        variation = np.random.uniform(-0.05, 0.05, len(df))
    elif i < 7:
        variation = np.random.uniform(-0.1, 0.1, len(df))
    else:
        variation = np.random.uniform(-0.15, 0.15, len(df))
    
    # Ensure entropy stays in valid range (0-8)
    df[f'entropy_feature_{i+1}'] = np.clip(df['entropy'] + variation, 0, 8)

# Select only the needed columns for our model
final_columns = [f'entropy_feature_{i+1}' for i in range(num_features)] + \
                ['file_type_txt', 'file_type_doc', 'file_type_mp4', 'file_type_pdf', 'family_label']

# Create final dataset
final_df = df[final_columns]

# Save the formatted dataset
final_df.to_csv('formatted_entropy_dataset.csv', index=False)

# Print information about the dataset
print(f"Dataset shape: {final_df.shape}")
print("\nFeature columns:")
for column in final_df.columns:
    print(f"- {column}")

print("\nRansomware family mapping:")
for family, label in sorted(family_mapping.items(), key=lambda x: x[1]):
    print(f"- Family {label}: {family}")

print("\nRansomware family distribution:")
family_counts = df['family_label'].value_counts().sort_index()
for family, count in family_counts.items():
    print(f"- Family {family}: {count} samples")

print("\nDataset saved to formatted_entropy_dataset.csv")