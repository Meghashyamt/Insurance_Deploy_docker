#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[50]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#import plotly.express as px
import seaborn as sns
sns.set(style='whitegrid')
import os
import pickle
#from plotly.offline import init_notebook_mode, iplot
for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Read Data

# First let's read datas and take a look at the data we have

# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


print("Train")
print(train.info())
print(train.isnull().sum())
print("-----------------------------------------------------------")
print("Test")
print(test.info())
print(test.isnull().sum())


# We can see that both data sets have not null value.

# # Label Encoding

# In machine learning, we usually deal with datasets which contains multiple labels in one or more than one columns. These labels can be in the form of words or numbers. To make the data understandable or in human readable form, the training data is often labeled in words.
# Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form.

# In[7]:


train['Vehicle_Age']=train['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})
train['Gender']=train['Gender'].replace({'Male':1,'Female':0})
train['Vehicle_Damage']=train['Vehicle_Damage'].replace({'Yes':1,'No':0})

test['Vehicle_Age']=test['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})
test['Gender']=test['Gender'].replace({'Male':1,'Female':0})
test['Vehicle_Damage']=test['Vehicle_Damage'].replace({'Yes':1,'No':0})


# We replaced some values in the data sets with numerical values, as follows;
# 
# **Vehicle Age ->**
# * "<1 Year" = 0
# * "1-2 Year" = 1
# * ">2 Year" = 2
# 
# **Gender ->**
# * "Female" = 0
# * "Male" = 1
# 
# **Vehicle Damage ->**
# * "No" = 0
# * "Yes" = 1

# In[8]:


train.head()


# # Correlation

# In[9]:


plt.figure(figsize=(12,12))
sns.heatmap(train.corr(),annot=True, fmt=".3f")


# We can see that the most influencing factors for Response are Vehicle_Damage and Previously_Insured, followed by Vehicle_Age and Policy_Sales_Channel.

# # An Overview Of The Data Set

# In[10]:


sns.countplot(train.Response)


# In[11]:


count_1 = train[train["Response"] == 1].value_counts().sum()
totalResponse = train["Response"].value_counts().sum()
print("The percentage of positive response in train data is :", round(count_1*100/totalResponse),"%")


# In[12]:


train.groupby(['Response','Vehicle_Age','Vehicle_Damage']).size()


# Most of the vehicles of customers with response 1 are between the ages of 1-2 and their vehicles are damaged.

# In[13]:


sns.countplot(x='Previously_Insured',data=train,hue='Response')


# Customers who were previously insured tend not to be interested. We can think that the reason for this is that their previous insurance agreement has not expired yet.

# In[14]:


print("Most used channel:")
print((train['Policy_Sales_Channel'].value_counts()))


# In[15]:


psc_notinterested=(train.loc[train['Response'][train['Response']==1].index.values])['Policy_Sales_Channel']
sns.distplot(psc_notinterested)
plt.title("Distribution of Policy Sales Channel for customers that were interested")
plt.show()


# The most used sales channels are 152, 26 and 124. The best channel that results in customer interest is 152.

# # Model Building - DecisionTreeClassifier

# Importing Libraries

# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Decision trees are one of the algorithms often used in the solution of classification problems. Its purpose is to create a model that estimates the value of a variable by extracting simple rules from data properties and learning these rules.

# First, I delete the "id" column as it will not contribute to model training.

# In[17]:


train.drop(columns="id", inplace=True, errors="ignore")


# I define the "Response" column to y and the other columns to X.

# In[18]:


X = train[train.columns[:-1]]
y = train[train.columns[-1]]


# We will divide our data into 4 variables;
# The x_train and y_train variables for training, x_test and y_test variables to test the model at the end of the training.
# 
# The test_size parameter specifies what percentage of the data set should be reserved for testing.

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Let's divide our data into 5 clusters with the K-Means algorithm.

# In[20]:


k_means = KMeans(n_clusters = 5,init='k-means++',random_state=0) 
clusters = k_means.fit(X) 
X['clusters'] = clusters.labels_


# In[21]:


k_means.labels_


# In[24]:


#fig=px.bar(X.groupby('clusters').count().reset_index(),x='clusters',y='Gender')
#fig.show()


# We see that most of the customers are gathered in cluster 2.

# In[25]:


cluster_2 = X[X['clusters']== 2]
cluster_2
model = ExtraTreesClassifier()
model.fit(X,y)
plt.figure(figsize=(8,6))
important_features = pd.Series(model.feature_importances_,index = cluster_2.columns)
important_features.nlargest(11).plot(kind = "bar")
plt.show()


# I choose customers in cluster 2 and used ExtraTreesClassifier () to find the most important features. So we can have an idea why customers are gathering more in cluster 2.

# In[26]:


#Creating a Model
tree_classifier = DecisionTreeClassifier()
#Building a relationship by looking at x_train and y_train data
tree_classifier.fit(X_train, y_train) 


# By sending X_test data to the predict function, we get a predict result.

# In[27]:


predictions = tree_classifier.predict(X_test)
predictions


# In[28]:


plot_confusion_matrix(tree_classifier,X_test, y_test)


# On this graph we can see how many of our predictions are correct.
# 
# * The bottom right corner is the number of values we guessed to be 1 and are actually 1, so it is True Positive.
# * The bottom left corner is the number of values we guessed to be 0 but are actually 1, so it is False Negative. 
# * The top right corner is the number of values we guessed to be 1 but are actually 0, so it is False Positive.
# * The top left corner is the number of values we guessed to be 0 and are actually 0, so it is True Negative.

# **Let's check the accuracy of the model.**

# **accuracy_score =** Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right.

# In[29]:


accuracy_score(y_test, predictions)


# **precision_score =**  It shows how many of the values we guess as Positive are actually Positive.

# In[30]:


precision_score(y_test, predictions)


# **recall_score =** It is a metric that shows how many of the operations we need to predict positive.

# In[31]:


recall_score(y_test, predictions)


# **f1_score =** The F1 Score value shows us the harmonic mean of the Precision and Recall values.

# In[32]:


f1_score(y_test, predictions)


# Let's try different models

# # RandomForestClassifier

# RandomForestClassifier generate multiple decision trees. When it will produce a result, the average value in these decision trees is taken and the result is produced.

# In[33]:


rf_classifier = RandomForestClassifier(random_state=1)
rf_classifier.fit(X_train, y_train)


# In[34]:


rf_predictions = rf_classifier.predict(X_test)
rf_predictions


# In[35]:


accuracy_score(y_test,rf_predictions), precision_score(y_test,rf_predictions), recall_score(y_test,rf_predictions), f1_score(y_test, rf_predictions)


# # KNeighborsClassifier

# The purpose of the K Nearest Neighbors algorithm, which is a classification algorithm, is to classify our data sets and then place the data whose class is unknown to the closest class.The number of elements to be looked at in the algorithm's work is determined by a K value. When a value comes, the distance between the value is calculated by taking the nearest K number of elements. The Euclidean function is generally used in the distance calculation. After the distance is calculated, it is sorted and the corresponding value is assigned to the appropriate class.

# In[36]:


KNN = KNeighborsClassifier(n_neighbors=11, metric='minkowski', p = 2)
KNN.fit(X_train, y_train)


# In[37]:


KNN_predictions = KNN.predict(X_test)
KNN_predictions


# In[38]:


accuracy_score(y_test,KNN_predictions), precision_score(y_test,KNN_predictions), recall_score(y_test,KNN_predictions), f1_score(y_test,KNN_predictions)


# # BaggingClassifier

# A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions to form a final prediction.

# In[39]:


b_classifier = BaggingClassifier()
b_classifier.fit(X_train, y_train)


# In[40]:


b_predictions = b_classifier.predict(X_test)
b_predictions


# In[41]:


accuracy_score(y_test,b_predictions), precision_score(y_test,b_predictions), recall_score(y_test,b_predictions), f1_score(y_test,b_predictions)


# # Logistic Regression Classifier

# A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions to form a final prediction.

# In[42]:


log_classifier = LogisticRegression()
log_classifier.fit(X_train, y_train)


# In[43]:


log_predictions = log_classifier.predict(X_test)
log_predictions


# In[44]:


accuracy_score(y_test,log_predictions)


# In[ ]:





# Compare the accuracy scores in all the models : 

# In[45]:


print("DecisionTreeClassifier Accuracy = ",accuracy_score(y_test, predictions))
print("RandomForestClassifier Accuracy = ",accuracy_score(y_test,rf_predictions))
print("KNeighborsClassifier Accuracy = ",accuracy_score(y_test,KNN_predictions))
print("BaggingClassifier Accuracy = ",accuracy_score(y_test,b_predictions))
print("Logistic Regression Classifier Accuracy = ",accuracy_score(y_test,log_predictions))


# It seems that KNeighborsClassifier has the best accuracy score. So I am going to use this model on submission.csv

# # File submission

# In[46]:


responses = KNN.predict(test[test.columns[1:]])


# In[47]:


submission = pd.DataFrame(data = {'id': test['id'], 'Response': responses})
submission.to_csv('submission.csv', index = False)
submission.head()


# In[51]:


# Saving model to disk
pickle.dump(rf_classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[ ]:





# In[ ]:




