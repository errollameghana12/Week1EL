###                             WEEK-2                         ###


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load the cleaned dataset from Week 1
df = pd.read_csv("Cleaned_Electric_Vehicle_Data.csv")

print("✅ Cleaned Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns)
# Select relevant features
features = ['Model Year', 'Make', 'Electric Vehicle Type', 'Base MSRP']
target = 'Electric Range'

df = df[features + [target]]
df.dropna(inplace=True)

print("\nSelected Columns for Modeling:")
print(df.head())
le_make = LabelEncoder()
le_type = LabelEncoder()

df['Make'] = le_make.fit_transform(df['Make'])
df['Electric Vehicle Type'] = le_type.fit_transform(df['Electric Vehicle Type'])

print("\n✅ Encoded Categorical Columns:")
print(df.head())
X = df[['Model Year', 'Make', 'Electric Vehicle Type', 'Base MSRP']]
y = df['Electric Range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n✅ Data prepared and saved successfully for model training!")
