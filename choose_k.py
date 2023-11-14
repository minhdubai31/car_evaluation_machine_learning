import pandas as pd

# Sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Prevent creating __pycache__ directories
import sys
sys.dont_write_bytecode = True

# Some utils function
from resources.utils import *


# Read dataset
car_data = pd.read_csv('resource/car_evaluation.csv', delimiter=',')



# Convert data's string values to numeric values
car_data = car_data_to_numeric(car_data)


# Split data to X and y for training
X = car_data.drop(columns='class')
y = car_data['class']


# Variables store model's test scores
f1 = []
k = []
# Number of tests
num_of_tests = 25


for i in  range(num_of_tests):
    # Split data using hold-out method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0)
    
    ### K Nearest Neighbors algorithm
    for i in range (1, 26):

        knn_model = KNeighborsClassifier(n_neighbors=i*2-1)
        knn_model.fit(X_train, y_train)
        # Predict test data
        y_pred = knn_model.predict(X_test)
        # Save f1 score
        f1.append(round(f1_score(y_true=y_test, y_pred=y_pred, average="micro")*100,3))
        k.append(i*2-1)


for i in range (1, 26):
    count = 0
    k_f1 = 0
    for index, each in enumerate(k):
        if each == i*2-1:
            k_f1 += f1[index]
            count += 1
    print(f"k = {i*2-1}, f1 = {round(k_f1/count, 3)}")
    



