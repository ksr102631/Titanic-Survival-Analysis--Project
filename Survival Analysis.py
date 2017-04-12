# Import the Pandas library

import pandas as pd

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())

 
 train.describe()
                        
 # Dimensions  
 
 train.shape()
 
# How many people in your training set survived the disaster with the Titanic? 
# To see this, you can use the value_counts() method in combination with standard bracket notation to select a single column of a DataFrame:

# In numbers
train.Survived.value_counts()

# In percentages
train.Survived.value_counts(normalize=True)

# To dive in a little deeper we can perform similar counts and percentage calculations on subsets of the Survived column. 
# For example, maybe gender could play a role as well? 
# You can explore this using the .value_counts() method for a two-way comparison on the number of males and females that survived



# Passengers that survived vs passengers that passed away
print(train.Survived.value_counts())

# As proportions
print(train.Survived.value_counts(normalize=True))

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == "male"].value_counts())

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == "female"].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == "male"].value_counts(normalize=True))

# Normalized female survival
print(train["Survived"][train["Sex"] == "female"].value_counts(normalize=True))


train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0
print(train["Child"])

# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))


# Create a copy of test: test_one
test_one = test

# Initialize a Survived column to 0
test["Survived"] = 0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test["Survived"][test["Sex"] == "female"] = 1

print(test_one.Survived)



# Import the Numpy library

import numpy as np

# Import 'tree' from scikit-learn library

from sklearn import tree

## Fill in the missing values

train.Age = train["Age"].fillna(train["Age"].median()) 

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S") 

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

#Print the Sex and Embarked columns
print(train.Sex)
print(train.Embarked)



# Print the train data to see the available features
print(train)

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))


# Impute the missing value with the median
test.Fare[152] = test.Fare.median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set and print them.
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)

## class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)[source]

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare","SibSp","Parch","Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two,target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two,target))


 
# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest        
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))


#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.score(features_forest, target))

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print(my_forest.feature_importances_)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
