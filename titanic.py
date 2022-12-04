#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# In[5]:


pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 16)
pd.set_option('precision', 2)

train = pd.read_csv('C://Users//RAM JIWAN//PycharmProjects//MachineLearningProject//MlProjectTitanic//train.csv')
test =pd.read_csv('c://users//RAM JIWAN//PycharmProjects//MachineLearningProject//MlProjectTitanic//test.csv')

print(train.describe()[:])
print("\n")


# In[6]:


train.describe()


# 

# In[7]:


train.columns


# In[8]:


train.head(10)


# In[9]:


train.sample(5)


# In[10]:


train.dtypes


# In[11]:


train.describe(include ="all")


# In[12]:


pd.isnull(train).sum()


# In[13]:


sbn.barplot(x ='Sex',y ='Survived',data =train)


# In[14]:


train


# In[15]:


train['Survived']


# In[16]:


train['Sex']=='female'


# In[17]:


train['Survived'][train['Sex']=='female']


# In[18]:


train['Survived'][train['Sex']=='female'].value_counts()


# In[19]:


train['Survived'][train['Sex']=='female'].value_counts(normalize=True)


# In[20]:


(train['Survived'][train['Sex']=='female'].value_counts(normalize=True)[1]*100)


# In[21]:


(train['Survived'][train['Sex']=='male'].value_counts(normalize=True)[1]*100)


# In[22]:


sbn.barplot(x ='Pclass',y ='Survived',data =train)


# In[23]:


(train['Survived'][train['Pclass']==1].value_counts(normalize=True)[1]*100)


# In[24]:


(train['Survived'][train['Pclass']==2].value_counts(normalize=True)[1]*100)


# In[25]:


(train['Survived'][train['Pclass']==3].value_counts(normalize=True)[1]*100)


# In[26]:


(train['Survived'][train['Pclass']==1].value_counts())


# In[27]:


sbn.barplot(x ='SibSp',y ='Survived',data = train)


# In[28]:


(train['Survived'][train['SibSp']==0].value_counts(normalize =True)[1]*100)


# In[29]:


(train['Survived'][train['SibSp']==1].value_counts(normalize =True)[1]*100)


# In[30]:


(train['Survived'][train['SibSp']==2].value_counts(normalize=True)[1]*100)


# In[31]:


sbn.barplot(x ='Parch' ,y ='Survived' ,data  = train)


# In[32]:


(train['Survived'][train['Parch']==0].value_counts(normalize =True)[1]*100)


# In[33]:


(train['Survived'][train['Parch']==1].value_counts(normalize =True)[1]*100)


# In[34]:


(train['Survived'][train['Parch']==2].value_counts(normalize =True)[1]*100)


# In[35]:


(train['Survived'][train['Parch']==3].value_counts(normalize = True)[1]*100)


# In[36]:


#Age features

train["Age"] =train['Age'].fillna(-0.5)
test["Age"] = test['Age'].fillna(-0.5)


# In[37]:


bins =   [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sbn.barplot(x="AgeGroup", y="Survived", data=train)


# In[38]:


train


# In[39]:


train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))


# In[40]:


train


# In[41]:


(train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)


# In[42]:


(train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)


# In[43]:


sbn.barplot(x ='CabinBool',y ='Survived',data =train)


# In[44]:


#cleaning the data
test.describe(include ='all')


# In[45]:


#we will drop the cabin feature 

train = train.drop(['Cabin'],axis =1)
test = test.drop(['Cabin'],axis =1)


# In[46]:


#we will also drop the ticket feature since it's unlikely to useful for any information

train =train.drop(['Ticket'],axis =1)
test  =test.drop(['Ticket'],axis =1)


# In[47]:


#Embarked Feature
#now we need to fill in the missing values in the Embarked feature

print( "Number of people embarking in Southampton (S):" ,  )
southampton = train[train["Embarked"] == "S"].shape[0]
print( southampton  )


# In[48]:


print( "Number of people embarking in Cherbourg (C):" ,  )
cherbourg = train[train["Embarked"] == "C"].shape[0]
print( cherbourg  )


# In[49]:


print( "Number of people embarking in Queenstown (Q):" ,  )
queenstown = train[train["Embarked"] == "Q"].shape[0]
print( queenstown  )


# In[50]:


#most of the people are embarked from soutnampton so we will fill the missing value with "S"
train = train.fillna({'Embarked':"S"})


# In[51]:


train


# In[52]:


#now we will cleaning the age feature
#create combine group of both data sets

combine =[train,test]
combine[0]
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(', ([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'] )
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
       ['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'],
        'Rare')

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train


# In[53]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).count()


# In[54]:


#map each of the title group to a nymerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
print( "\n\nAfter replacing title with neumeric values.\n"  )
train  


# In[55]:


mr_age = train[train["Title"] == 1]["AgeGroup"].mode()
mr_age


# In[56]:


miss_age = train[train["Title"] == 2]["AgeGroup"].mode()
miss_age


# In[57]:


mrs_age = train[train["Title"] == 3]["AgeGroup"].mode()
mrs_age


# In[58]:


master_age = train[train["Title"] == 4]["AgeGroup"].mode()
master_age


# In[59]:


royal_age = train[train["Title"] == 5]["AgeGroup"].mode()
royal_age


# In[60]:


rare_age = train[train["Title"] == 6]["AgeGroup"].mode()
rare_age


# In[61]:


train.head()


# In[62]:


print( "\n\n********   train[AgeGroup][0] :  \n\n"  )
for x in range(10) :
      print(train["AgeGroup"][x] )


# In[63]:


age_title_mapping = {1: "Young Adult", 2: "Student",
                3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[  train["Title"][x]  ]
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
print( "\n\nAfter replacing Unknown values from AgeGroup column : \n"  )
train  


# In[64]:


age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,
               'Student': 4, 'Young Adult': 5,
               'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
print()
train


# In[65]:


#drop the age column and Name column
train = train.drop(['Age'], axis=1)
test = test.drop(['Age'], axis=1)
print( "\n\nAge column and name column droped."  )
train


# In[66]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)
train


# In[67]:


#embarked mapping
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
train.head()


# In[68]:


for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x]
        test["Fare"][x] = round(train[ train["Pclass"] == pclass ]["Fare"].mean(), 2)

train['FareBand'] = pd.qcut(train['Fare'], 4,
                            labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4,
                           labels = [1, 2, 3, 4])


train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)

print( "\n\nFare column droped\n"  )
train


# In[69]:


from sklearn.model_selection import train_test_split
input_predictors = train.drop(['Survived', 'PassengerId'], axis=1)
ouptut_target = train["Survived"]
x_train, x_val, y_train, y_val=train_test_split(
    input_predictors, ouptut_target, test_size = 0.20, random_state = 7)


# In[70]:


#apply the logistic regression model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
" Accuracy of LogisticRegression : ", acc_logreg  


# In[71]:


#Creating Submission Result File
#***********************************

#It is time to create a submission.csv file which includes our predictions for test data

ids = test['PassengerId']
predictions = logreg.predict(test.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

print( "All survival predictions done." )
print( "All predictions exported to submission.csv file." )

print( output )


# In[72]:


sample =pd.read_csv('submission.csv')
sample.head()


# In[73]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_val, y_pred)
print(confusion_matrix)


# In[74]:


from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




