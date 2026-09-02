import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(r"D:\INTERNSHIP PROJECT\dataset\train.csv")

# -----------------------------
# Basic Dataset Information
# -----------------------------
print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== COLUMN NAMES ==========\n")
print(df.columns)

print("\n========== DATASET INFORMATION ==========\n")
df.info()

print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========\n")
print(df.duplicated().sum())


# ============================================
# EDA (Exploratory Data Analysis)
# ============================================

# 1 Placement Status Distribution
plt.figure(figsize=(6,5))
sns.countplot(x='Placement_Status', data=df)
plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.show()

# 2 Gender Distribution
plt.figure(figsize=(6,5))
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# 3 Degree Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Degree', data=df)
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 4 Branch Distribution
plt.figure(figsize=(12,5))
sns.countplot(x='Branch', data=df)
plt.title("Branch Distribution")
plt.xlabel("Branch")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 5 CGPA Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['CGPA'], bins=20, kde=True)
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Frequency")
plt.show()

# 6 Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()