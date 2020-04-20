import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#Importing the dataset
df = pd.read_csv('Iris.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, 5].values

# Encoding dependent variable
le = LabelEncoder()
Y = le.fit_transform(Y)

# Split dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=0)

#Create the Decision Tree model and fitting the data
decision_tree_classifier = DecisionTreeClassifier(random_state=0)
decision_tree_classifier.fit(X, Y)

# Predicting Test set results
Y_pred = decision_tree_classifier.predict(X_test)
Y_pred

#Calculating and printing the model accuracy
print(f'The accurancy score is:{accuracy_score(Y_pred,Y_test)*100}%')
