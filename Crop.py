# Crop Recommendation System with Visuals - Final Version

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
url = "Crop_recommendation.csv"
df = pd.read_csv(url)

print("✅ Dataset Loaded Successfully!")
print(df.head())

# 2. Basic Info
print("\n📄 Dataset Info:")
print(df.info())

# 3. Visualize Correlation Heatmap
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title("📊 Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 4. Crop Distribution
plt.figure(figsize=(12, 5))
sns.countplot(data=df, y='label', order=df['label'].value_counts().index, palette='viridis')
plt.title("🌾 Crop Label Distribution")
plt.xlabel("Count")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()

# 5. Features and Target Split
X = df.drop('label', axis=1)
y = df['label']

# 6. Encode Target Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# 8. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Predict and Evaluate
y_pred = model.predict(X_test)

print("\n🎯 Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 10. Confusion Matrix Heatmap
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=False, cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("🧠 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 11. Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names, palette='cubehelix')
plt.title("📈 Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# 12. Test with New Input
sample_input = np.array([[90, 42, 43, 20.8, 82, 6.5, 200]])
prediction = model.predict(sample_input)
predicted_crop = le.inverse_transform(prediction)

print("\n🌱 Recommended Crop for given values:", predicted_crop[0])
