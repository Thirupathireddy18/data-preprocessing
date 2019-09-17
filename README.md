# data-preprocessing
#claeaning
#loading required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#Loading dataset
health = pd.read_csv("E:\\Datasets\\DataSet\\health.csv")
health.head(4)

#Changing Categorical to Binary
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Lencoder= LabelEncoder()
health.iloc[:, 1:2]= Lencoder.fit_transform(health.iloc[:, 1:2])
health.iloc[:, 4:5]= Lencoder.fit_transform(health.iloc[:, 4:5])

#get dummies for multiple-categories present in a column with the help of pandas
#it will include all the categorical variables and convert to dummies ,no  need to select the column
health= pd.get_dummies(health)
health.head(2)


#Checking null values     
print(health.isnull().sum())

#droping the null values
print(health.dropna().head(2))

#Repalcing NUll values with strategy
from sklearn.preprocessing import Imputer
imputer_mean = Imputer(missing_values='NaN', strategy="mean", axis=0)
imputer_most_frequent = Imputer(missing_values='NaN', strategy="most_frequent", axis=0)
health.iloc[:, 5:6]= imputer_mean.fit_transform(health.iloc[:, 5:6])
health.iloc[:, 1:2]= imputer_most_frequent.fit_transform(health.iloc[:, 1:2])

#standard scalar
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
health.iloc[:, 5:6]= sc_x.fit_transform(health.iloc[:, 5:6])

#splitting the data
x= health .iloc[:, 5:6].values
y= health.iloc[:, 1:5].values 
