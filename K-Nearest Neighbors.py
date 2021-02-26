#!/usr/bin/env python
# coding: utf-8

# # KNN on Customer Dataset 
# 
# 
# <h1>Table of contents</h1>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="#1">KNN Theory</a></li>
#         <li><a href="#2">KNN Algorithm</a></li>
#         <li><a href="#3">Calculating the distance</a></li>
#         <li><a href="#4">Dataset</a></li>
#         <li><a href="#5">Label-encoding using Scikit learn library</a></li>
#         <li><a href="#6">Data Visualization and Analysis</a></li>
#         <li><a href="#7">Train Test Split</a></li>
#         <li><a href="#8">Claasification</a></li>
#         <li><a href="#12">What about other K</a></li>
#         <li><a href="#13">Plot model accuracy for different number of neighbors</a></li>
#     </ol>
# </div>
# <br>
# <hr>
# 
# <div id="1">
#     <h2>KNN Theory</h2>
# </div>
# 
# K-Nearest Neighbors is an algorithm for supervised learning. Where the data is trained with data points corresponding to their classification. Once a point is to be predicted, it takes account the K nearest points to it to determines it's classification 
# 
# 
# <div id="2">
#     <h2>KNN Algorithm</h2>
# </div>
# 
# 1. Pick a value of K
# 
# 2. Calculate the distance of unknown case from all the cases(Euclidian, Manhattan, Minkowski or Weighted)
# 
# 3. Select the K-observations in the training data that are nearest to the unknown data point 
# 
# 4. Predict the response of the unknown data point using the most popular response value from the K-Nearest Neighbors
# 
# ### Here's an visualization of the K-Nearest Neighbors algorithm.
# 
# 
# <img src="https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2019/03/How-does-KNN-Algorithm-work-KNN-Algorithm-In-R-Edureka-528x250.png">
# 
# 
# Here we have two data points of Class A and B. We want to predtict what the red circle (test data point) is. If we consider a k value of 3. We will obtain a predtiction of Class A. Yet if we consider a k value of 6, we will obtain a prediction of Class B. 
# 
# Looking at from this perspective we can say that it is important to consider the value of K. In this diagram it considers the K Nearest Neighbors when it predicts the classification of the test point (Red Circle).

# <div id="3">
#     <h2>Calculating the distance</h2>
# </div>
# 
# To calculate the distance between two points with one point being the unknown case and the other being the data you have in your dataset. To calculate there are several ways, we are gonna use euclidean distance. 
# 
# ### Euclidean distance formule
# 
# <img src="https://i.stack.imgur.com/RtnTY.jpg">
# 
# let's consider you have 6 columns and 5 rows. The last column is the prediction columns have different Class labels.
# 
# Step 1. 
# 
# **Subtraction**
# 
# Subtract each attribute(column) from row 1 with the attribute from row 2.
# 
# Example = (2-3) = -1
# 
# Step 2.
# 
# **Exponention**
# 
# After the subtracting column 1 from row 1, with column 1 from row 2, we will get squared root. 
# 
# **Important**
# Results are always positive 
# 
# Example = (2-3)** 2 = (-1)** 2 = 1
# 
# Step 3.
# 
# **Sum**
# 
# Once you are done performing step 1 and 2 on all the attribute(column) from row 1 with the attribute from row 2. Sum all the results.
# 
# Example = (2-3)** 2 + (2-3)** 2 + (4-5)** 2 + (5-6)** 2 + (2-3)** 2 = 5
# 
# 
# Step 4.
# 
# **Square root**
# 
# After step 3, we will square root the result and that the euclidean distance from line 1 to line 2.
# 
# Perform the same steps with respect to other lines and you will have the euclidean distance from line 1 to all other lines. Once that is done. We will check which is the class that most appears. The class which appears more time will be the class that we will use to classify the unknown case. 

# ### Import Libraries

# In[102]:


import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <div id="4">
#     <h2>Dataset</h2>
# </div>
# 
# 
# Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. It is a classification problem. That is, given the dataset,  with predefined labels, we need to build a model to be used to predict class of a new or unknown case. 
# 
# The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns. 
# 
# The target field, called **custcat**, has four possible values that correspond to the four customer groups, as follows:
#   1- Basic Service
#   2- E-Service
#   3- Plus Service
#   4- Total Service
# 
# Our objective is to build a classifier, to predict the class of unknown cases. We will use a specific type of classification called K nearest neighbour.
# 
# You can find this dataset in IBM SPSS.
# 

# ### Load the dataset

# In[14]:


df = pd.read_excel("telco2.xlsx")
df.head()


# In[113]:


#making a copy of the dataset
df1 = df.copy()


# ### Data Exploration

# In[114]:


df1.shape


# In[115]:


#Checking for any missing values 
df1.isna().sum()


# We can see that the column logtoll, logequi, logcard and logwire are having missing values. As we are not gonna be using these columns for over analysis we will drop them.

# In[116]:


df1 = df1.drop(["loglong","logtoll","logequi","logcard","logwire"],axis =1)
df1.head()


# In[117]:


df1.isna().values.sum()


# There are no missing values 

# In[118]:


df1.info()


# In[119]:


df1.describe(include="all")


# In[120]:


df1.fillna(0)


# In[121]:


df1.isna().values.sum()


# For our analysis we will take the columns tenure, age, marital, address, income, ed, employ, retire, gender, reside and custcat

# In[122]:


df1 = df1.drop(["tollfree","equip","callcard","wireless","longmon","tollmon","equipmon","cardmon","wiremon","longten","tollten","equipten","cardten","wireten","multline","voice","pager","internet","callid","callwait","forward","confer","ebill","lninc","churn"],axis=1)


# In[123]:


df1.head()


# <div id="5">
#     <h2>Label-encoding using Scikit learn library </h2>
# </div>

# In[124]:


#Import label encoder

from sklearn import preprocessing 

#laabel_encoder object knows how to understand word labels

label_encoder = preprocessing.LabelEncoder()

#Encode labels in column region, marital, ed, retire, gender, custcat

df1['region'] = label_encoder.fit_transform(df1['region'])
df1['marital'] = label_encoder.fit_transform(df1['marital'])
df1['ed'] = label_encoder.fit_transform(df1['ed'])
df1['retire'] = label_encoder.fit_transform(df1['retire'])
df1['gender'] = label_encoder.fit_transform(df1['gender'])
df1['custcat'] = label_encoder.fit_transform(df1['custcat'])


# In[125]:


df1.head()


# Now that we have the proper data. We can do Data visualization and Analysis on it.

# <div id="6">
#     <h2>Data Visualization and Analysis</h2>
# </div>

# Let's see how many of each class is in our dataset 

# In[126]:


df1["custcat"].value_counts()


# 0. **Basic-service = 266**
# 
# 1. **E-Service Customer = 217**
# 
# 2. **Plus Service = 281**
# 
# 3. **Total Service = 236**

# In[127]:


df1.hist(column ='income',bins=50)


# In[128]:


ax = sns.countplot(y=df1['custcat'], data=df1)
plt.title('Distribution of custcat')
plt.xlabel('Number of Axles')

total = len(df1['custcat'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
plt.show()


# 0. **Basic-service = 266**
# 
# 1. **E-Service Customer = 217**
# 
# 2. **Plus Service = 281**
# 
# 3. **Total Service = 236**

# We can see from the countplot the Plus Service is been used more then other services.

# ## Feature set

# In[129]:


df1.columns


# In[130]:


X = df1[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]


# In[133]:


y = df['custcat'].values
y[0:5]


# ## Normalize Data
# 
# Data Standardization give data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on distance of cases:

# In[135]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# <div id = "7">
#     <h2>Train Test Split</h2>
# </div>
# 
# Out of Sample Accuracy is the percentage of correct predictions that the model makes on data that that the model has NOT been trained on. Doing a train and test on the same dataset will most likely have low out-of-sample accuracy, due to the likelihood of being over-fit.
# 
# It is important that our models have a high, out-of-sample accuracy, because the purpose of any model, of course, is to make correct predictions on unknown data. So how can we improve out-of-sample accuracy? One way is to use an evaluation approach called Train/Test Split.
# Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set. 
# 
# This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. It is more realistic for real world problems.

# In[136]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =4)
print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)


# <div id="8">
#     <h2> Claasification </h2>
#     </div>
#     
# K-Nearest neighbor (KNN)

# In[137]:


#import library 
from sklearn.neighbors import KNeighborsClassifier


# <div id="9">
#     <h2>Training </h2>
#     </div>

# In[175]:


#Let's start the algorithm with k = 5 and keep on updating to see which one is close 

k = 9
#Train model and Predict 
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# <div id="10">
#     <h2> Predicting</h2>
#     </div>

# In[164]:


yhat = neigh.predict(X_test)
yhat[0:5]


# <div id="11">
#     <h2> Accuracy evaluation </h2>
#     </div>
# 
# In multilabel classification, **accuracy classification score** is a function that computes subset accuracy. This function is equal to the jaccard_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.

# In[174]:


from  sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test,yhat))


# <div id="12">
#     <h2> What about other K</h2>
#     </div>
# 
# K in KNN, is the number of nearest neighbors to examine. It is supposed to be specified by the User. So, how can we choose right value for K?
# The general solution is to reserve a part of your data for testing the accuracy of the model. Then chose k =1, use the training part for modeling, and calculate the accuracy of prediction using all samples in your test set. Repeat this process, increasing the k, and see which k is the best for your model.
# 
# We can calculate the accuracy of KNN for different Ks.
# 

# In[173]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# <div id="13">
#     <h2> Plot model accuracy for different number of neighbors </h2>
#     </div>

# In[171]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[172]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

