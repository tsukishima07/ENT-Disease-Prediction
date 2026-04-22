import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt



# Step 1: Load Dataset
df = pd.read_csv("cleaned_dataset.csv", encoding="utf-8")
print(f"Original dataset shape: {df.shape}")


# Step 2: Clean Dataset
# Remove completely empty columns
df = df.dropna(axis=1, how='all')

# Drop 'all_symptoms' column if it exists (we'll recreate it)
if 'all_symptoms' in df.columns:
    df = df.drop('all_symptoms', axis=1)

# Fill missing symptom cells with placeholder
symptom_cols = df.columns[1:]  # all columns except 'Disease'
df[symptom_cols] = df[symptom_cols].fillna('missing')

# Merge all symptoms into a list
df['all_symptoms'] = df[symptom_cols].apply(lambda x: [i for i in x if i != 'missing'], axis=1)

# Remove rare diseases (<3 samples)
disease_counts = df['Disease'].value_counts()
rare_diseases = disease_counts[disease_counts < 3].index.tolist()
df = df[~df['Disease'].isin(rare_diseases)]
print(f"After removing rare diseases: {df.shape}")

# Balance dataset by oversampling minority diseases
dfs = []
max_count = df['Disease'].value_counts().max()
for disease, group in df.groupby('Disease'):
    dfs.append(resample(group, replace=True, n_samples=max_count, random_state=42))
df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Balanced dataset shape: {df_balanced.shape}")


# Step 3: Prepare Features
# Create and fit MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_balanced['all_symptoms'])
y = df_balanced['Disease']

print(f"\nTotal unique symptoms: {len(mlb.classes_)}")
print(f"Feature matrix shape: {X.shape}")


# Step 4: Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 5: Train Model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
print("\nTraining RandomForest model...")
model.fit(X_train, y_train)


# Step 6: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n Test Accuracy: {accuracy:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Step 7: Save Model & Symptom List
joblib.dump(model, "model.pkl")
joblib.dump(mlb.classes_.tolist(), 'symptom_list.pkl')  #  Fixed: Save symptom list as Python list
print("\n Model and symptom list saved successfully!")
print(f" Saved {len(mlb.classes_)} unique symptoms to symptom_list.pkl")


# Step 8: Optional Visualization
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10, 5))
plt.bar(range(15), importances[indices], color="teal", alpha=0.7)
plt.xticks(range(15), [mlb.classes_[i] for i in indices], rotation=45, ha="right")
plt.title("Top 15 Important Symptoms for Disease Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\n Feature importance plot saved as 'feature_importance.png'")
plt.show()