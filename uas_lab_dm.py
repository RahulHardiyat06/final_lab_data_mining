#!/usr/bin/env python
# coding: utf-8

# In[108]:


#Importing Libraries 
#basics and Visualization
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt



#ML libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 



#metrics
from statistics import mean
from sklearn.metrics import accuracy_score as score
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import classification_report, confusion_matrix




#Ignore Warning 
import warnings as wrn
wrn.filterwarnings('ignore')


# In[109]:


url = "http://nrvis.com/data/mldata/pima-indians-diabetes.csv"

# buat nama header
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', ' DiabetesPedigreeFunction', 'Age', 'ClassVariable']

# baca dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)
dataset.head()


# In[110]:


dataset.info()


# In[111]:


# Variabel independen
x = dataset.drop(["ClassVariable"], axis = 1)
x.head()


# In[112]:


# Variabel dependen
y =dataset.ClassVariable.astype('category')
y.head()


# In[113]:


dataset.isnull().sum()


# In[114]:


dataset['BMI'].fillna(dataset['BMI'].median(), inplace=True)
dataset['Glucose'].fillna(dataset['Glucose'].median(), inplace=True)
dataset['BloodPressure'].fillna(dataset['BloodPressure'].median(), inplace=True)
dataset['SkinThickness'].fillna(dataset['SkinThickness'].median(), inplace=True)
dataset['Insulin'].fillna(dataset['Insulin'].median(), inplace=True)

dataset.describe()


# In[115]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 123)


# In[116]:


#Feature Scaling
 
scaler = StandardScaler()  
scaler.fit(x_train)
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)


# In[117]:


#KNN

classifier = KNeighborsClassifier(n_neighbors=5) 
classifier.fit(x_train, y_train)


# In[118]:


#Prediction n=5
y_pred = classifier.predict(x_test)
print(y_pred)


# In[119]:


y_actual1=y_test.tolist()
y_pred1=y_pred.tolist()


# In[120]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[121]:


error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40): 
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(x_train, y_train)
 pred_i = knn.predict(x_test)
 error.append(np.mean(pred_i != y_test))


# In[122]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', 
 markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value') 
plt.xlabel('K Value') 
plt.ylabel('Mean Error')
plt.show()


# In[123]:


#Training and Prediction

classifier1 = KNeighborsClassifier(n_neighbors=39) 
classifier1.fit(x_train, y_train)


# In[124]:


#Prediction
y_pred2 = classifier1.predict(x_test)
print(y_pred2)


# In[125]:


y_actual2=y_test.tolist()
y_pred2=y_pred.tolist()


# In[126]:


#Evaluate
print(confusion_matrix(y_actual2, y_pred2))


# In[127]:


clf = KNeighborsClassifier(n_neighbors=39)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print('Accuracy in percent = ',score(pred, y_test)*100)


# In[128]:


#Classification and prediction
#SVM
model = SVC(kernel='linear')
model.fit(x_train, y_train)
pred = model.predict(x_test)
print('Accuracy in percent = ',score(pred, y_test)*100)


# In[129]:


model = SVC(C=2, kernel='rbf', gamma='scale')
model.fit(x_train, y_train)

pred = model.predict(x_test)
print (round(accuracy_score(y_test, pred)*100,2))


# In[130]:


clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print('Accuracy in percent = ',score(pred, y_test)*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




