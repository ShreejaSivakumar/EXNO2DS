# EXNO2DS
# NAME: SHREEJA R S
# REF.NO :  25017561

# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
STEP 1: Import Required PackageS
```
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
```
```
%matplotlib inline
sns.set_style("whitegrid")
```
Load the Dataset
```
import pandas as pd

df = pd.read_csv("titanic_dataset.csv")
print(df.head())

```
Basic Information of Dataset
```
df.info()
```
```
df.describe()
```

STEP 2: Handling Missing Values
```
df.isnull().sum()
```
```
# Fill missing Age using Median
df['Age'] = df['Age'].fillna(df['Age'].median())
```
```
# Fill missing Embarked using Mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```
```
# Drop Cabin column if it exists
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])
df.isnull().sum()
```
STEP 3: Boxplot to Analyze Outliers

```
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age")
plt.show()
```
```
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()
```

STEP 4: Removing Outliers using IQR Method

```
import pandas as pd

# Assuming df is already loaded
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
df = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]

print("Dataset Shape after removing outliers:", df.shape)

```

STEP 5: Countplot for Categorical Data
```
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Count of Survived")
plt.show()
```
```
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Count")
plt.show()
```

STEP 6: Displot for Univariate Distribution

```
sns.displot(df['Age'] , kde =  True)
plt.title('Distribution of Age')
plt.show()
```
```
sns.displot(df['Fare'], kde= True)
plt.title('Distribution o fare')
plt.show()
```
STEP 6: Displot for Univariate Distribution

```
crosstab = pd.crosstab(df['Pclass'], df ['Survived'])
crosstab
```

STEP 8: Heatmap Representation

```
plt.figure(figsize=(6,4))
sns.heatmap(crosstab, annot=True, cmap="coolwarm" , fmt='d')
plt.title("Heatmap of Pclass vs Survived")
plt.show()
```
```
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

 All the above code cells and their corresponding outputs (tables and graphical representations) are included in this notebook

# RESULT
  Thus, the Exploratory Data Analysis was successfully performed on the given Titanic dataset. 
Missing values were handled appropriately, outliers were detected using boxplot and removed 
using the IQR method. Categorical data was analyzed using countplot, numerical distributions 
were studied using displot, and relationships between variables were examined using cross 
tabulation and heatmap representation.


