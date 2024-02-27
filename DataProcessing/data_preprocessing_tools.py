# Feature are called independent variables
# Independent Variables: The variable that are not affected by the other variables are called independent variables. For example age of a person, is an independent variable, two person’s born on same date will have same age irrespective of how they lived. We presume that while independent variables are stable and cannot be manipulated by some other variable, they might cause a change in other variables,
# Dependent Variables: The variables which depend on other variables or factors. We expect these variables to change when the independent variables, upon whom they depend, undergo a change. They are the presumed effect. For example let us say you have a test tomorrow, then, your test score is dependent upon the amount of time you studied, so the test score is a dependent variable, and amount of time independent variable in this case.
# with the help of independet variable we will predict dependant variables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#--------------------------------------
print("-----------------------------pima-indians-diabetes-database start------------------------------")

# Load the dataset
#dataset = datasets.load_p
dataset = pd.read_csv('pima-indians-diabetes.csv')
#dataset = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Identify missing data (assumes that missing data is represented as NaN)
print(dataset.columns)
missing_data = dataset.isnull().sum()
# Print the number of missing entries in each column
print(f"Missing data : \n {missing_data}")
# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy = "mean")
# Fit the imputer on the DataFrame
imputer.fit(X[:, :-1])
# Apply the transform to the DataFrame
X[:, :-1]=imputer.transform(X[:, :-1])
#Print your updated matrix of features
print(X)
print("-----------------------------pima-indians-diabetes-database End------------------------------")
#--------------------------------------

print("-----------------------------------titanic csv start---------------")
# Load the dataset
df = pd.read_csv('titanic.csv')

# Identify the categorical data
X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

categorical_features = ['Sex', 'Embarked', 'Pclass']

# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),categorical_features)], remainder='passthrough')

# Apply the fit_transform method on the instance of ColumnTransformer
X = ct.fit_transform(df)

# Convert the output into a NumPy array
X = np.array(X)

# Use LabelEncoder to encode binary categorical data
le = LabelEncoder()
y = le.fit_transform(df['Survived'])

# Print the updated matrix of features and the dependent variable vector
print(X)
print(y)

print("-----------------------------------titanic csv End---------------")

print("-----------------------------------------------------------------")
print("-----------------------------------Data csv Start---------------")
dataset = pd.read_csv("Data.csv")
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc
#dataframe.iloc[row, column]
X= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values
print(X)
print(y)
#iris = datasets.load_iris()
#print(iris)
#dataset = pd.DataFrame(iris.data)
#print(dataset)

###############################################################
# Taking Care of Missing Data for Numerical Data
###############################################################
# Simpleimputer is used to populate Missing Values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# fit () — This method goes through the training data, calculates the parameters (like mean (μ) and standard deviation (σ) in StandardScaler class ) and saves them as internal objects.
imputer.fit(X[:, 1:3])
#transform() — The parameters generated using the fit() method are now used and applied to the training data to update them.
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

###############################################################
# Encoding Categorical Data
###############################################################
# Encoding Independent Variables

ct = ColumnTransformer(transformers=[('encoding',OneHotEncoder(),[0])], remainder='passthrough')
X=np.array(ct.fit_transform(X))

# Encoding Dependent Variables
le = LabelEncoder()
y=le.fit_transform(y)
print(X)
print(y)
print(type(X))

# Feature scaling needs to be after the data is split into Tranining and Test data set
# Because while doing feature scaling we shouldn't include test data into consideration , as it will cause information leakage.
# The primary reason we split before scaling is to prevent data leakage. Data leakage happens when information from outside the training dataset is used to create the model

# RandomState instance or None, default=None
# Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("-----------------------------------------------------------------")
print("-----------------------------------Data csv End---------------")
