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
