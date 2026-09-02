import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv(r"D:\INTERNSHIP PROJECT\dataset\train.csv")

print(df.head())
print(df.isnull().sum())

# ---------------------------
# Handle Missing Values
# ---------------------------
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ---------------------------
# Remove Duplicates
# ---------------------------
print("Duplicates before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates after:", df.duplicated().sum())

# ---------------------------
# Label Encoding (FIXED)
# ---------------------------
encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le   # save encoder for each column

# Save encoders
pickle.dump(encoders, open("encoders.pkl", "wb"))

print(df.head())

# ---------------------------
# Split Features & Target
# ---------------------------
X = df.drop("Placement_Status", axis=1)
y = df["Placement_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(X_train.shape)
print(X_test.shape)

# ---------------------------
# Feature Scaling
# ---------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Step 3 Completed Successfully")
df = pd.read_csv(r"D:\INTERNSHIP PROJECT\dataset\train.csv")

print(df.columns.tolist())