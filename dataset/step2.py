# ==========================================================
# STEP 2 : EXPLORATORY DATA ANALYSIS (EDA)
# Project : Smart Campus Placement Prediction System
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(r"D:\INTERNSHIP PROJECT\dataset\train.csv")

sns.set_style("whitegrid")

# ==========================================================
# 1. Internships Distribution
# ==========================================================
plt.figure(figsize=(7,5))
sns.countplot(x='Internships', data=df)
plt.title("Internships Distribution")
plt.xlabel("Number of Internships")
plt.ylabel("Number of Students")
plt.show()

# ==========================================================
# 2. Projects Distribution
# ==========================================================
plt.figure(figsize=(7,5))
sns.countplot(x='Projects', data=df)
plt.title("Projects Distribution")
plt.xlabel("Projects")
plt.ylabel("Number of Students")
plt.show()

# ==========================================================
# 3. Coding Skills Distribution
# ==========================================================
plt.figure(figsize=(7,5))
sns.histplot(df['Coding_Skills'], bins=10, kde=True)
plt.title("Coding Skills Distribution")
plt.xlabel("Coding Skills")
plt.ylabel("Frequency")
plt.show()

# ==========================================================
# 4. Communication Skills Distribution
# ==========================================================
plt.figure(figsize=(7,5))
sns.histplot(df['Communication_Skills'], bins=10, kde=True)
plt.title("Communication Skills Distribution")
plt.xlabel("Communication Skills")
plt.ylabel("Frequency")
plt.show()

# ==========================================================
# 5. Aptitude Test Score Distribution
# ==========================================================
plt.figure(figsize=(7,5))
sns.histplot(df['Aptitude_Test_Score'], bins=10, kde=True)
plt.title("Aptitude Test Score Distribution")
plt.xlabel("Aptitude Test Score")
plt.ylabel("Frequency")
plt.show()

# ==========================================================
# 6. Soft Skills Rating Distribution
# ==========================================================
plt.figure(figsize=(7,5))
sns.histplot(df['Soft_Skills_Rating'], bins=10, kde=True)
plt.title("Soft Skills Rating Distribution")
plt.xlabel("Soft Skills Rating")
plt.ylabel("Frequency")
plt.show()

# ==========================================================
# 7. Certifications Distribution
# ==========================================================
plt.figure(figsize=(7,5))
sns.countplot(x='Certifications', data=df)
plt.title("Certifications Distribution")
plt.xlabel("Certifications")
plt.ylabel("Number of Students")
plt.show()

# ==========================================================
# 8. Backlogs Distribution
# ==========================================================
plt.figure(figsize=(7,5))
sns.countplot(x='Backlogs', data=df)
plt.title("Backlogs Distribution")
plt.xlabel("Backlogs")
plt.ylabel("Number of Students")
plt.show()

# ==========================================================
# 9. Placement Status vs Gender
# ==========================================================
plt.figure(figsize=(7,5))
sns.countplot(x='Gender', hue='Placement_Status', data=df)
plt.title("Placement Status vs Gender")
plt.show()

# ==========================================================
# 10. Placement Status vs Degree
# ==========================================================
plt.figure(figsize=(8,5))
sns.countplot(x='Degree', hue='Placement_Status', data=df)
plt.xticks(rotation=45)
plt.title("Placement Status vs Degree")
plt.show()

# ==========================================================
# 11. Placement Status vs Branch
# ==========================================================
plt.figure(figsize=(12,5))
sns.countplot(x='Branch', hue='Placement_Status', data=df)
plt.xticks(rotation=45)
plt.title("Placement Status vs Branch")
plt.show()

# ==========================================================
# 12. CGPA vs Placement (Boxplot)
# ==========================================================
plt.figure(figsize=(7,5))
sns.boxplot(x='Placement_Status', y='CGPA', data=df)
plt.title("CGPA vs Placement Status")
plt.show()

# ==========================================================
# 13. Correlation Heatmap
# ==========================================================
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()