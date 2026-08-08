import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

class Knn_classification:
    def __init__(self, n_neighbours):
        # Initialize the number of neighbors and storage for training data
        self.n_neighbours = n_neighbours
        self.X = None
        self.y = None 

    def fit(self, X, y):
        # Store the training features and labels
        self.X = X
        self.y = y

    def accuracy(self, y_test, y_pred):
        # Compute and return prediction accuracy
        return accuracy_score(y_test, y_pred)    

    def predict(self, X_test):
        y_pred = []
        for test_x in X_test:
            dist_array = []
            # Calculate Euclidean distance between test point and all training points
            for train_x in self.X: 
                dist = np.sum(np.square(test_x - train_x))
                dist_array.append(dist)
                
            # Ensure shape compatibility for sorting
            temp_y = self.y.reshape(len(dist_array))
            
            # Sort distances and rearrange corresponding labels together
            d, y = (list(t) for t in zip(*sorted(zip(dist_array, temp_y))))
            
            # Select the labels of the K-nearest neighbors
            y_labels = y[:self.n_neighbours]
            
            # Find the most common class among the nearest neighbors
            b = Counter(y_labels)
            class_name = b.most_common()[0][0]
            
            # Append the predicted class to the results list
            y_pred.append(class_name)
            
        return y_pred