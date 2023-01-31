#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Read data into pandas dataframe
df = pd.read_csv('predictive_maintenance.csv')
print(df.info())


# In[3]:


# Eliminate UDI and Product ID columns
df = df.drop('UDI',axis='columns')
df = df.drop('Product ID',axis='columns')
df.sample(5)


# In[4]:


# Visualize Failure occurrence on data set
title='Failure Count'
xlabel = 'Failure'
ylabel = '# of machines'
legend = ['Failure','No failure']
df['Target'].value_counts().plot(kind = 'bar',legend = legend,xlabel=xlabel,ylabel=ylabel,title=title,color='r')


# In[5]:


# Visualize Machine quaity incidence in data set
title='Machine Quality'
xlabel = 'Quality'
ylabel = '# of machines'
df['Type'].value_counts().plot(kind = 'bar',legend = legend,xlabel=xlabel,ylabel=ylabel,title=title,color='r')


# In[6]:


# Visualize failure types in data set
s = [failure for failure in df['Failure Type'] if failure != 'No Failure']
s = pd.DataFrame(s)

title='Failure Type'
xlabel = 'Failure Type'
ylabel = '# of machines'
d = s.value_counts()
d.plot(kind = 'bar',legend = legend,xlabel=xlabel,ylabel=ylabel,title=title,color='r')


# In[9]:


# Visualize trends in seaborn pair plot
import seaborn as sns
sns.pairplot(df, hue = 'Failure Type')


# In[10]:


# Import logistic regression model usinf sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# label Enconder
enc = preprocessing.LabelEncoder()

# make copy of original data frame for modification
df1 = df.copy(deep=True)

# cconverting Failure Type and Type columns to numerical categories
df1['Failure Type']=enc.fit_transform(df1['Failure Type'])
df1['Type']=enc.fit_transform(df1['Type'])
df1.sample(5)


# In[11]:


#Mapping encoded values to actual string values
f = df['Type'].value_counts()
ff = df1['Type'].value_counts()
print(f)
print('-----------')
print(ff)
print('------------------')

f = df['Failure Type'].value_counts()
ff = df1['Failure Type'].value_counts()
print(f)
print('-----------')
print(ff)


# In[15]:


# Model - Target = Failure Type

# separate training data
X = df1[['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']]
y = df1['Failure Type']

# Split training data into training and test (70% training 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

#Logistic regression
logreg = LogisticRegression(solver='lbfgs',max_iter=10000)
logreg.fit(X_train,y_train)
logreg.get_params(deep=True)

#Return the mean accuracy on the given test data and labels.
score = logreg.score(X_test,y_test)
print('Mean accuracy on given test data: {}'.format(score))

# Return model parameters
print('\n Model parameters: {}'.format(logreg.get_params(deep=True)))


# In[16]:


# Funtion that returns actual string value of encoded prediction
def uncoded_prediction(prediction):
    pred = []
    if prediction == 1:
        pred.append("No Failure")
    if prediction == 2:
        pred.append('Overstrain Failure')
    if prediction == 3:
        pred.append('Power Failure')
    if prediction == 4:
        pred.append('Random Failures')
    if prediction == 5:
        pred.append('Tool Wear Failure')
    if prediction == 0:
        pred.append('Heat Dissipation Failure')
    return pred


# In[17]:


#Testing model - making a few predictions

import warnings
warnings.filterwarnings('ignore')

x = np.array([[3,70,500,700,700,400]])
prediction = logreg.predict(x)[0]
un_pred = uncoded_prediction(prediction)

proba_prediction = logreg.predict_proba(x)

print(un_pred)
print('Prediction probability {}'.format(np.max(proba_prediction)))

print('-------------')

x = np.array([[1,303.3,311.4,1497,46,30]])
prediction = logreg.predict(x)[0]
un_pred = uncoded_prediction(prediction)

proba_prediction = logreg.predict_proba(x)

print(un_pred)
print('Prediction probability {}'.format(np.max(proba_prediction)))


# In[ ]:




