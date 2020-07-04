#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[7]:


# Using the pandas Excel class to raed the excel file load the excel file
data_set = pd.ExcelFile('Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx')
sheet = 0  # Sheet counter

#Seperating the sheets and saving them as CSV file
# for sheet_name in data_set.sheet_names:
#     if sheet <= len(data_set.sheet_names):
#         print(f'Reading {data_set.sheet_names[sheet]} to a DataFrame....')
#         df = pd.read_excel(data_set, sheet_name)
#         name = f'sheet{sheet}.csv'
#         df.to_csv(name)
#         print()
#         print(
#             f'Saved {data_set.sheet_names[sheet]} as a CSV file with the name {name}')
#         sheet += 1


# In[26]:


def existing_employee():
    existing_employees = pd.read_csv("Existing employees.csv")
    existing_employees_count = len(existing_employees["Emp ID"])
    left_company = []
    
    while existing_employees_count != 0:
        left_company.append("YES")
        existing_employees_count -= 1
    
    print("Before adding the new left_company column")
    print(existing_employees.head)
    existing_employees["left_company"] = left_company
    print("After adding the new left_company column")
    return existing_employees
    
existing_employees = existing_employee()
existing_employees.head()


# In[28]:


def left_employee():
    left_employees = pd.read_csv("Employees that left.csv")
    left_employees_count = len(left_employees["Emp ID"])
    left_company = []
    
    while left_employees_count != 0:
        left_company.append("NO")
        left_employees_count -= 1
        
    left_employees["left_company"] = left_company
    print(left_employees.head)
    return left_employees
employees_that_left = left_employee()
employees_that_left.head()


# In[10]:


frames = [existing_employees, employees_that_left]
total_employees = pd.concat(frames)
total_employees.to_csv("total_employee_dataset.csv")
total_employees


# In[11]:


#Selecting the feature and target dataset
X = total_employees[["satisfaction_level","last_evaluation","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","dept","salary"]]
Y = total_employees["left_company"]


# In[12]:


# Encoding categorical data AND Encoding independent variable
X = pd.get_dummies(X)
X.head()


# In[29]:



# Encoding dependent variable
le = LabelEncoder()
Y = le.fit_transform(Y)


# In[84]:


# Split dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.40,random_state=0)

#Create the Random Forest model and fitting the data
classifier  = RandomForestClassifier(n_estimators=11, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)


# In[85]:


# Predicting Test set results
Y_pred = classifier.predict(X_test)
Y_pred


# In[86]:


#Calculating and printing the model accuracy
print(f'The accurancy score is:{accuracy_score(Y_pred,Y_test)*100}%')


# In[18]:


feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')


# In[81]:


#Reselecting the feature and target dataset in order to predict employee that is prone to leave
X1 = total_employees[["satisfaction_level","last_evaluation","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","dept","salary"]]
Y1 = total_employees["left_company"]

# Split dataset into train and test set
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1,Y1,test_size=0.40,random_state=0)

employee_prone_to_leave = pd.DataFrame(X_test1)
employee_prone_to_leave["left_company"]  = Y_test1
employee_prone_to_leave["predicted"] = Y_pred
employee_prone_to_leave


# In[82]:


# Predicting employees that will leave the company
leaving = employee_prone_to_leave[(employee_prone_to_leave["predicted"] == 1) & (employee_prone_to_leave["left_company"] != "YES") ]
leaving.to_csv("Employees prone to leave")


# In[ ]:




