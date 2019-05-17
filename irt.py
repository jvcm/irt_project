import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import SGDRegressor, LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class IRTModel:

    def __init__(self, models = []):
        self.models = models
        self.irt_matrix = []
        return

    def setModels(self, models = []):
        self.models = models
        return

    def fit(self, X_train = [], y_train = []):
        self.y_mean = y_train.mean()
        for model in self.models:
            model.fit(X_train, y_train)
        return

    def irtMatrix(self, X_test = [], y_test = [], noise_std = 0.0,\
         normalize = False, base_models = True, name = 'dataset', rd = 42):
        # IRT matrix shape
        n = len(y_test)
        m = len(self.models)

        X_test_ = X_test
        if type(X_test_) == np.ndarray:
            X_test_ = pd.DataFrame(X_test,)

        # Noise generated
        # noise_train = np.random.normal(loc=0.0, scale= noise_std, size= len(y_train))
        # noise_test = np.random.normal(loc=0.0, scale= noise_std, size= len(y_test))

        # Apply noise
        # y_train = y_train + noise_train
        # y_test = y_test + noise_test
        X_test_['noise'] = 0
        
        # Fit regression models
        # self.fit(X_train= X_train, y_train= y_train)
        
        names = list(map(lambda x: type(x).__name__, self.models))
        indexes = y_test.index.values

        irt_matrix = np.zeros((m,n))
        errors = np.zeros((m,n))

        error_df = pd.DataFrame()

        for i, model in enumerate(self.models):
            y_pred = model.predict(X_test)
            error_df[i] = (y_test - y_pred)
            errors[i, :] = np.absolute(y_test - y_pred)
            if normalize == True:
                errors[i, :] = errors[i, :]/ np.absolute(y_test - y_test.mean())
            
            for j, instance in enumerate(errors[i, :]):
                # f = 1/(1+np.exp(-instance)) # função sigmoide
                # irt_matrix[i, j] = 2 - 2*f # caso utilize a função sigmoide
                f = np.clip(1/(1 + instance), 1e-4, 1-1e-4)
                irt_matrix[i, j] = f
        self.irt_matrix = pd.DataFrame(data= irt_matrix, index= names, columns = indexes).T
        error_df.columns = names
        if base_models:
            self.irt_matrix['Average'] = 0.5 + np.random.rand(len(self.irt_matrix))*0.0001
            self.irt_matrix['Optimal'] = self.irt_matrix.apply(func = max, axis = 1)
            self.irt_matrix['Worst'] = self.irt_matrix.apply(func = min, axis = 1)
        
        # Writing files
        X_test_.to_csv(path_or_buf= './beta_irt/xtest_'+ name + '_s' + str(n) + '_f' + str(int(noise_std)) + '_sd' + str(rd) +'.csv', index= False, encoding='utf-8')
        self.irt_matrix.to_csv(path_or_buf= './beta_irt/irt_data_' + name + '_s' + str(n) + '_f' + str(int(noise_std)) + '_sd' + str(rd) +'.csv', index= False, encoding='utf-8')
        error_df.to_csv(path_or_buf= './beta_irt/errors_' + name + '_s' + str(n) + '_f' + str(int(noise_std)) + '_sd' + str(rd) +'.csv', index= False, encoding='utf-8')
        return

def beta_irt(thetai, deltaj, aj):
    p1 = ((deltaj)/(1 - deltaj))** aj
    p2 = ((thetai)/(1 - thetai))** -aj
    den = 1 + p1 * p2
    return 1/den

def main():
    return

if __name__ == "__main__":
    main()
