import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# KNN Class
class KNearestNeighbor:

    # Constructor for class which sets up k, X_train and y_train
    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        print ('k:', k)

    # Calls compute_distance function to find Euclidean distances
    # Then uses function "predict_presence" with distances to predict outcome
    def predict(self, X_test):
        distances = self.compute_distance(X_test)
        return self.predict_presence(distances)

    # Computes Euclidean distance between X_test and X_train points and returns results
    def compute_distance(self, X_test):
        # Sets num_test to the size of X_test
        num_test = X_test.shape[0]
        # Set num_train to the size of X_train
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        # Euclidean distance for all points
        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sqrt(np.sum((X_test[i, :] - self.X_train[j, :])**2))
        print distances.shape[0]
        return distances

    # Predicts presence of heart disease (1,2,3,4), 0 = no presence, by using Euclidean distances
    def predict_presence(self, distances):
        # num_test is the size of distances which is the same size as the testing set.
        num_test = distances.shape[0]
        # Sets y_pred to array of zeros the same size of the testing set / distances
        y_pred = np.zeros(num_test)
        # Loop to find presence for each instances in testing data
        for i in range(num_test):
            # Sorts distances to find closest points
            y_indices = np.argsort(distances[i, :])
            # Only looks at K neighbors in y_indices
            k_nearest_neighbors = self.y_train[y_indices[:self.k]].astype(int)
            print('KNN for i', k_nearest_neighbors, i)
            # Counts most common in array y_pred
            y_pred[i] = np.argmax(np.bincount(k_nearest_neighbors))
            print ('y has been classified as:', y_pred[i])
        return y_pred


# Main
if __name__ == "__main__":
    # Sets column name for dataset with only the following columns
    # headers = ["Age", "Sex", "CP", "Trestbps", "Chol", "Num"]
    # # Sets data to simpler dataset
    # heart_data = pd.read_csv("processed.va.data.csv", names=headers)

    # Sets column names for "Complete dataset"
    headers2 = ["Age", "Sex", "CP", "Trestbps", "Chol", "FBS", "Restecg", "Thalach", "Exang",
               "Oldpeak", "Slope", "Ca", "Thal", "Num"]
    # Sets data to "Complete" dataset
    heart_data = pd.read_csv("Complete.processed.va.data.csv", names=headers2)

    # Drops rows with missing value
    # heart_data.drop(heart_data[heart_data['Trestbps'] == '?'].index, inplace=True)
    # heart_data.drop(heart_data[heart_data['Chol'] == '?'].index, inplace=True)
    # heart_data.drop(heart_data[heart_data['FBS'] == '?'].index, inplace=True)
    # heart_data.drop(heart_data[heart_data['Restecg'] == '?'].index, inplace=True)
    # heart_data.drop(heart_data[heart_data['Thalach'] == '?'].index, inplace=True)
    # heart_data.drop(heart_data[heart_data['Exang'] == '?'].index, inplace=True)
    # heart_data.drop(heart_data[heart_data['Oldpeak'] == '?'].index, inplace=True)
    # heart_data.drop(heart_data[heart_data['Slope'] == '?'].index, inplace=True)
    # heart_data.drop(heart_data[heart_data['Ca'] == '?'].index, inplace=True)
    # heart_data.drop(heart_data[heart_data['Thal'] == '?'].index, inplace=True)

    # This helps make the KNN become more accurate as there are only 2 levels of Presence
    heart_data.Num.replace(1, 0, inplace=True)
    heart_data.Num.replace(2, 0, inplace=True)
    heart_data.Num.replace(3, 1, inplace=True)
    heart_data.Num.replace(4, 1, inplace=True)

    # Replaces all '?' with 0
    heart_data.replace(to_replace="?", value=0, inplace=True)

    # Sets X to the first 3 columns
    X = heart_data.iloc[:, :13].astype(float).values

    # Sets y to the last column
    y = heart_data.iloc[:, -1].values

    # Splits data to training and testing data from dataset 10% is used for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Normalizes data in X
    sc_X = preprocessing.StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)

    # Sets k
    k = 5
    # Prints size of training data
    print X_train.shape[0]
    # initializes KNN class
    KNN = KNearestNeighbor(k, X_train, y_train)
    # Runs predict function in KNN class
    y_pred = KNN.predict(X_test)
    # Calculates accuracy of KNN by counting how many times y_pred is equal to y_test
    # and dividing it by length of y_test.
    accuracy = (np.sum(y_pred == y_test) / len(y_test)) * 100
    print('These are the predicted y:', y_pred)
    print ('These are the y_test data:', y_test)
    print('Accuracy using KNN from scratch: {:.2f}%'.format(accuracy))

    # Using this section to compare results from the KNN above to the KNN in scikit library
    print ('Using library')

    classifier = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2)
    classifier = classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    print ('Accuracy: {:.2f}%'.format(accuracy))