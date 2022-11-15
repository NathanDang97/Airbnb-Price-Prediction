from typing import Tuple
import numpy as np
from numpy.linalg import inv
import time


class Model:

    ID_DICT = {"NAME": "N/A", "ID": "N/A", "EMAIL": "N/A"}

    def __init__(self):
        self.theta = None
        self.is_train_data_processed = False
        self.features_to_drop = set()
        self.best_lambda = 0.0

    # helper method that adds a column of 1's to a matrix
    def add_column(self, X: np.array) -> np.array:
        return np.vstack((np.ones((X.shape[0],)), X.T)).T

    # helper method that removes constant features
    def constant_features(self, X):
        const_features = set()
        for i in range(X.shape[1]):
            if np.std(X[:, i], axis=0) == 0:
                const_features.add(i)
        return const_features

    # helper method that removes duplicate features
    def duplicate_features(self, X: np.array) -> np.array:
        dup_features = set()
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                if np.array_equal(X[:,i], X[:,j]):
                    dup_features.add(j)
        return dup_features

    def preprocess(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        ###############################################
        ####      add preprocessing code here      ####
        ###############################################

        time_start = time.time()

        if not self.is_train_data_processed:
            const_features = self.constant_features(X)
            dup_features = self.duplicate_features(X)
            self.features_to_drop.update(const_features)
            self.features_to_drop.update(dup_features)
            self.is_train_data_processed = True

        print("Dropping duplicate and constant features...")
        X = np.delete(X, list(self.features_to_drop), axis=1)
        print("New shape of features: ", X.shape)
        print("New shape of labels: ", y.shape)

        time_end = time.time()
        print("Total data preprocessing time: ", f"{np.round((time_end - time_start) /60, 2)} minutes")
        print()
 
        return X, y

    # helper method that run linear regression with L2-regularisation using normal equation
    def normal_equation_with_reg(self, X_train: np.array, y_train: np.array, lambda_=0.0) -> np.array:   

        lambda_matrix = lambda_ * np.identity(X_train.shape[1])
        lambda_matrix = np.insert(lambda_matrix, 0, 0, axis=0)
        lambda_matrix = np.insert(lambda_matrix, 0, 0, axis=1)
        X_train= self.add_column(X_train)

        theta = inv(X_train.T @ X_train + lambda_matrix) @ (X_train.T @ y_train)

        return theta

    # grid searching method for the best regularising parameter
    def search_lambda(self, X_train_full: np.array, y_train_full: np.array) -> float:
        
        curr_lambda = 1.0
        final_lambda = 1.0
        min_mse = float('inf')
        avg_curr_mse = 0.0
        
        for i in range(100):
            mse_runs = []
            for i in range(5): # 5-fold cross-validation
                rows = X_train_full.shape[0]
                random_indices = np.random.choice(rows, size=round(0.8*X_train_full.shape[0]), replace=False)
                X_train_curr = np.delete(X_train_full, random_indices, axis=0)
                X_val_curr = X_train_full[random_indices, :]
                y_train_curr = np.delete(y_train_full, random_indices, axis=0)
                y_val_curr = y_train_full[random_indices, :]

                current_theta = self.normal_equation_with_reg(X_train_curr, y_train_curr, curr_lambda)
                X_val_curr = self.add_column(X_val_curr)
                y_pred_curr = np.dot(X_val_curr, current_theta)
                current_mse = np.square(np.subtract(y_val_curr, y_pred_curr)).mean()
                mse_runs.append(current_mse)
            
            avg_curr_mse = np.mean(mse_runs)
            
            if avg_curr_mse < min_mse:
                min_mse = avg_curr_mse
                final_lambda = curr_lambda
            curr_lambda += 0.1
        
        return final_lambda

    def train(self, X_train: np.array, y_train: np.array):
        """
        Train model with training data
        """
        ###############################################
        ####   initialize and train your model     ####
        ###############################################

        time_start = time.time()

        print("MODEL: Linear Regression with L2-Regularisation")

        print("Grid searching for the best regularising parameter...")
        self.best_lambda = self.search_lambda(X_train, y_train)

        print("Calculating the final coefficients...")
        self.theta = self.normal_equation_with_reg(X_train, y_train, self.best_lambda)

        time_end = time.time()
        print("Total training time: ", f"{np.round((time_end - time_start) /60, 2)} minutes")
        print()


    def predict(self, X_val: np.array) -> np.array:
        """
        Predict with model and given feature
        """
        ###############################################
        ####      add model prediction code here   ####
        ###############################################

        X_val = self.add_column(X_val)
        return np.dot(X_val, self.theta)
