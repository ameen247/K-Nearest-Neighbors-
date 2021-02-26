# K-Nearest-Neighbors

KNN on Customer Dataset 

K-Nearest Neighbors is an algorithm for supervised learning. Where the data is trained with data points corresponding to their classification. Once a point is to be predicted, it takes account the K nearest points to it to determines it's classification 

KNN Algorithm

1. Pick a value of K

2. Calculate the distance of unknown case from all the cases(Euclidian, Manhattan, Minkowski or Weighted)

3. Select the K-observations in the training data that are nearest to the unknown data point.

4. Predict the response of the unknown data point using the most popular response value from the K-Nearest Neighbors

About the Dataset

Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. It is a classification problem. That is, given the dataset,  with predefined labels, we need to build a model to be used to predict class of a new or unknown case. 

The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns. 

The target field, called custcat, has four possible values that correspond to the four customer groups, as follows:
  1- Basic Service
  2- E-Service
  3- Plus Service
  4- Total Service

Our objective is to build a classifier, to predict the class of unknown cases. We will use a specific type of classification called K nearest neighbour.

You can find this dataset in IBM SPSS.

