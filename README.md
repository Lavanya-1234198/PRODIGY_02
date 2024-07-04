# PRODIGY_02

Titanic Dataset Analysis
Overview
This repository contains Python code for performing data cleaning and exploratory data analysis (EDA) on the Titanic dataset obtained from Kaggle. The analysis aims to derive insights into passenger demographics and survival rates.

Steps Taken
Data Cleaning
Handling Missing Values:
Filled missing values in 'Age' with the median age.
Filled missing values in 'Embarked' with the most common port.
Dropped the 'Cabin' column due to excessive missing values.
Dropped rows with missing values in 'Fare'.
Exploratory Data Analysis (EDA)
Summary Statistics:

Calculated summary statistics for numerical features like age and fare.
Visualizations:

Distribution of Age: Histogram showing the distribution of passenger ages.
Survival Rate by Sex: Bar plot comparing survival rates between male and female passengers.
Survival Rate by Passenger Class: Bar plot showing survival rates across different passenger classes.
Survival Rate by Embarkation Port: Bar plot illustrating survival rates based on embarkation ports.
Pairplot: Scatterplot matrix to explore relationships between numerical variables like 'Survived', 'Pclass', 'Age', and 'Fare'.
Tools Used
Python libraries:
Pandas
Seaborn
Matplotlib

SCRIPT:

#Task-2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/dbms/train (2).csv'
titanic_df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(titanic_df.head())

# Display the column names and data types
print("\nColumn names and data types:")
print(titanic_df.info())

# Check for missing values
print("\nMissing values in each column:")
print(titanic_df.isnull().sum())

# Data Cleaning
# Fill missing values in 'Age' with the median age
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common port
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column as it has too many missing values
titanic_df.drop(columns=['Cabin'], inplace=True)

# Drop rows with missing values in 'Fare' (if any)
titanic_df.dropna(subset=['Fare'], inplace=True)

# Check for missing values again
print("\nMissing values after cleaning:")
print(titanic_df.isnull().sum())

# Exploratory Data Analysis (EDA)
# Summary statistics
print("\nSummary statistics:")
print(titanic_df.describe())

# Visualization: Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(titanic_df['Age'], kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Visualization: Survival rate by Sex
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=titanic_df)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.show()

# Visualization: Survival rate by Pclass
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=titanic_df)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Visualization: Survival rate by Embarked
plt.figure(figsize=(10, 6))
sns.barplot(x='Embarked', y='Survived', data=titanic_df)
plt.title('Survival Rate by Embarkation Port')
plt.xlabel('Embarkation Port')
plt.ylabel('Survival Rate')
plt.show()

# Pairplot to see relationships between numerical variables
sns.pairplot(titanic_df[['Survived', 'Pclass', 'Age', 'Fare']])
plt.show()

Result:
![t21](https://github.com/Lavanya-1234198/PRODIGY_02/assets/174336088/11db0662-238f-4c12-98ba-36f898f3907a)

![t22](https://github.com/Lavanya-1234198/PRODIGY_02/assets/174336088/bb10f716-cb07-4422-a3dd-42c2108b46c3)

![t23](https://github.com/Lavanya-1234198/PRODIGY_02/assets/174336088/ae28b74a-67ad-4360-94d1-019183a62765)

![t24](https://github.com/Lavanya-1234198/PRODIGY_02/assets/174336088/1938380b-5bbf-4013-825d-54697867ecde)

Next Steps
Further analysis could include feature engineering to create new variables that might better predict survival.
Machine learning models could be applied to predict survival probabilities based on passenger features.
