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
<img width="602" height="332" alt="Screenshot 2026-03-10 223305" src="https://github.com/user-attachments/assets/11e46051-3e70-4677-b8fe-2cda03214171" />



Load the Dataset
```
import pandas as pd

df = pd.read_csv("titanic_dataset.csv")
print(df.head())

```
<img width="1364" height="848" alt="Screenshot 2026-03-10 223331" src="https://github.com/user-attachments/assets/d6aa2498-987a-4560-bf42-7c15f5b3d77a" />



Basic Information of Dataset
```
df.info()
```

<img width="1292" height="729" alt="Screenshot 2026-03-10 223536" src="https://github.com/user-attachments/assets/3647e493-739b-4278-805e-0e4d2bfc293a" />

```
df.describe()
```


STEP 2: Handling Missing Values
```
df.isnull().sum()
```
<img width="1412" height="988" alt="Screenshot 2026-03-10 223553" src="https://github.com/user-attachments/assets/aaedc781-abc6-4370-a639-db12d28e290d" />


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
<img width="1384" height="893" alt="Screenshot 2026-03-10 223612" src="https://github.com/user-attachments/assets/dbbb715a-ceb7-402c-ad3b-2ac17aed2ce9" />



STEP 3: Boxplot to Analyze Outliers

```
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age")
plt.show()
```
<img width="1275" height="924" alt="Screenshot 2026-03-10 223632" src="https://github.com/user-attachments/assets/a887ef6b-149f-49a4-aed2-1e1b510823be" />

```
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()
```
<img width="1281" height="919" alt="Screenshot 2026-03-10 223651" src="https://github.com/user-attachments/assets/515f0c45-aa8e-4e2f-af0d-395219b75b3c" />

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
<img width="1304" height="677" alt="Screenshot 2026-03-10 223707" src="https://github.com/user-attachments/assets/58d2c902-0870-404a-88d9-07535907ff75" />


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
<img width="1258" height="828" alt="Screenshot 2026-03-10 223726" src="https://github.com/user-attachments/assets/a8e6321d-b86e-403c-a845-4511d6a39be3" />

<img width="1291" height="857" alt="Screenshot 2026-03-10 223740" src="https://github.com/user-attachments/assets/36109ab0-69b1-4505-859f-f1938a58f750" />


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

<img width="1273" height="958" alt="Screenshot 2026-03-10 223753" src="https://github.com/user-attachments/assets/92404440-28ef-4171-a0dd-4a5862ae30a2" />

<img width="1286" height="956" alt="Screenshot 2026-03-10 223805" src="https://github.com/user-attachments/assets/6fd7df7b-63c9-48ee-8be1-b3c848711c17" />

STEP 6: Displot for Univariate Distribution

```
crosstab = pd.crosstab(df['Pclass'], df ['Survived'])
crosstab
```
<img width="1304" height="472" alt="Screenshot 2026-03-10 223824" src="https://github.com/user-attachments/assets/624ecd79-2044-4b5e-8bf0-dc2bd49175bc" />

STEP 7: Heatmap Representation

```
plt.figure(figsize=(6,4))
sns.heatmap(crosstab, annot=True, cmap="coolwarm" , fmt='d')
plt.title("Heatmap of Pclass vs Survived")
plt.show()
```
<img width="1281" height="837" alt="Screenshot 2026-03-10 223840" src="https://github.com/user-attachments/assets/2cf134c9-e225-4bde-894c-024826713eee" />

```
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```
<img width="1274" height="986" alt="Screenshot 2026-03-10 223902" src="https://github.com/user-attachments/assets/65a30fd9-018d-4f16-9861-7c98f99b53f4" />


 All the above code cells and their corresponding outputs (tables and graphical representations) are included in this notebook

# RESULT
  Thus, the Exploratory Data Analysis was successfully performed on the given Titanic dataset. 
Missing values were handled appropriately, outliers were detected using boxplot and removed 
using the IQR method. Categorical data was analyzed using countplot, numerical distributions 
were studied using displot, and relationships between variables were examined using cross 
tabulation and heatmap representation.


