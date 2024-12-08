import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print('[info] all necessary packaged imported successfully...')
data = pd.read_csv('Crop_recommendation.csv')
data

# Custom neon-themed color palette
neon_palette = ['#39FF14', '#FF073A', '#FFD700', '#1E90FF', '#DA70D6', '#00FFFF', '#FF1493']

# Violin plots for each column with custom colors
plt.figure(figsize=(16, 10))
for i, (column, color) in enumerate(zip(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], neon_palette)):
    plt.subplot(3, 3, i+1)
    sns.violinplot(y=data[column], color=color)
    plt.title(f'Violin plot of {column}', color=color)
plt.tight_layout()
plt.savefig('eda_1.png',bbox_inches='tight')
plt.show()

# Box plots for each column with custom colors
plt.figure(figsize=(16, 10))
for i, (column, color) in enumerate(zip(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], neon_palette)):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=data[column], color=color)
    plt.title(f'Box plot of {column}', color=color)
plt.tight_layout()
plt.savefig('eda_2.png',bbox_inches='tight')
plt.show()

# Pie plot for the distribution of the last column
crop_counts = data['label'].value_counts()

# Assigning a color for each crop type (reuse neon palette if there are more labels)
colors = neon_palette * (len(crop_counts) // len(neon_palette) + 1)
colors = colors[:len(crop_counts)]

plt.figure(figsize=(10, 10))
plt.pie(crop_counts, labels=crop_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Pie Chart of Crop Type Distribution')
plt.axis('equal')
plt.savefig('eda_3.png', bbox_inches='tight')
plt.show()

# Count plot for the distribution of the last column
plt.figure(figsize=(12, 8))
sns.countplot(x='label', data=data, palette=neon_palette)
plt.title('Count Plot of Crop Type Distribution', fontsize=16)
plt.xlabel('Crop Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_4.png', bbox_inches='tight')
plt.show()