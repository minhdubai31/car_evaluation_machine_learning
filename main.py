import pandas as pd
import numpy as np

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Prevent creating __pycache__ directories
import sys
sys.dont_write_bytecode = True

# Some utils function
from resource.utils import *


# Read dataset
car_data = pd.read_csv('resource/car_evaluation.csv', delimiter=',')


# Export pie charts displaying values distribution of each attribute (to image files)
for attribute in car_data.columns:
    export_values_distribution(
        column=car_data[attribute], 
        save_path='img/values_distribution/'+attribute+'.png',
        title='Phân bố giá trị trong cột \"'+attribute+'\"'
    )


# Convert data's string values to numeric values
car_data = car_data_to_numeric(car_data)


# Split data to X and y for training
X = car_data.drop(columns='class')
y = car_data['class']


# Export heatmap chart (to an image file)
export_heatmap(data=car_data, save_path='img/heatmap/car_heatmap.png')


# Variables store model's test scores
knn_f1_score = []
bayes_f1_score = []
randomforest_f1_score = []


# Number of tests
num_of_tests = 50


for i in  range(num_of_tests):
    # Split data using hold-out method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0)
    
    ### K Nearest Neighbors algorithm
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(X_train, y_train)
    # Predict test data
    y_pred = knn_model.predict(X_test)
    # Calculate f1 score
    knn_f1_score.append(round(f1_score(y_true=y_test, y_pred=y_pred, average="micro")*100,3))

    ### Bayes algorithm
    bayes_model = GaussianNB()
    bayes_model.fit(X_train, y_train)
    # Predict test data
    y_pred = bayes_model.predict(X_test)
    # Calculate f1 score
    bayes_f1_score.append(round(f1_score(y_true=y_test, y_pred=y_pred, average="micro")*100,3))

    ### Decision Tree Classifier algorithm
    randomforest_model = RandomForestClassifier()
    randomforest_model.fit(X_train, y_train)
    # Predict test data
    y_pred = randomforest_model.predict(X_test)
    # Calculate f1 score
    randomforest_f1_score.append(round(f1_score(y_true=y_test, y_pred=y_pred, average="micro")*100,3))    


# Export line chart displaying f1 scores of each model (to an image file)
export_line_chart(
    np.array(range(1, num_of_tests+1)), 
    [knn_f1_score, bayes_f1_score, randomforest_f1_score], 
    ["KNN", "Bayes", "Random Forest"], 
    ["b", "g", "r"],
    "img/line_chart/line.png"    
)


# Export feature importances chart (to an image file)
export_feature_importances(randomforest_model.feature_importances_, X.columns, 'img/feature_importances/feature_importances.png')


print(f"Giá trị F1 trung bình KNN = {round(sum(knn_f1_score)/len(knn_f1_score), 3)}")
print(f"Giá trị F1 trung bình Bayes = {round(sum(bayes_f1_score)/len(bayes_f1_score), 3)}")
print(f"Giá trị F1 trung bình Random Forest = {round(sum(randomforest_f1_score)/len(randomforest_f1_score), 3)}")

