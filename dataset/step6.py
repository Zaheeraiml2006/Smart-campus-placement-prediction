import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv(r"D:\INTERNSHIP PROJECT\dataset\train.csv")

# Create graph folder
os.makedirs("static/graphs", exist_ok=True)

sns.set_style("whitegrid")


# 1. Internships Distribution
plt.figure(figsize=(7, 5))
sns.countplot(x="Internships", data=df)
plt.title("Internships Distribution")
plt.xlabel("Number of Internships")
plt.ylabel("Number of Students")
plt.savefig("static/graphs/internships.png")
plt.close()


# 2. Projects Distribution
plt.figure(figsize=(7, 5))
sns.countplot(x="Projects", data=df)
plt.title("Projects Distribution")
plt.xlabel("Number of Projects")
plt.ylabel("Number of Students")
plt.savefig("static/graphs/projects.png")
plt.close()


# 3. Coding Skills
plt.figure(figsize=(7, 5))
sns.histplot(df["Coding_Skills"], bins=10, kde=True)
plt.title("Coding Skills Distribution")
plt.xlabel("Coding Skills")
plt.ylabel("Frequency")
plt.savefig("static/graphs/coding_skills.png")
plt.close()


# 4. Communication Skills
plt.figure(figsize=(7, 5))
sns.histplot(df["Communication_Skills"], bins=10, kde=True)
plt.title("Communication Skills Distribution")
plt.xlabel("Communication Skills")
plt.ylabel("Frequency")
plt.savefig("static/graphs/communication_skills.png")
plt.close()


# 5. CGPA vs Placement
plt.figure(figsize=(7, 5))
sns.boxplot(x="Placement_Status", y="CGPA", data=df)
plt.title("CGPA vs Placement Status")
plt.savefig("static/graphs/cgpa_placement.png")
plt.close()


# 6. Placement vs Gender
plt.figure(figsize=(7, 5))
sns.countplot(x="Gender", hue="Placement_Status", data=df)
plt.title("Placement Status vs Gender")
plt.savefig("static/graphs/gender_placement.png")
plt.close()


# 7. Placement vs Degree
plt.figure(figsize=(8, 5))
sns.countplot(x="Degree", hue="Placement_Status", data=df)
plt.xticks(rotation=45)
plt.title("Placement Status vs Degree")
plt.savefig("static/graphs/degree_placement.png")
plt.close()


# 8. Placement vs Branch
plt.figure(figsize=(10, 5))
sns.countplot(x="Branch", hue="Placement_Status", data=df)
plt.xticks(rotation=45)
plt.title("Placement Status vs Branch")
plt.savefig("static/graphs/branch_placement.png")
plt.close()


# 9. Correlation Heatmap
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("static/graphs/correlation.png")
plt.close()


print("All graphs created successfully!")