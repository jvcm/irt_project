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

    def irtMatrix(self, X_test = [], y_test = [], normalize = False, base_models = True):
        n = len(y_test)
        m = len(self.models)

        names = list(map(lambda x: type(x).__name__, self.models))
        indexes = y_test.index.values

        irt_matrix = np.zeros((m,n))
        errors = np.zeros((m,n))

        for i, model in enumerate(self.models):
            y_pred = model.predict(X_test)
            errors[i, :] = np.absolute(y_test - y_pred)
            if normalize == True:
                errors[i, :] = errors[i, :]/ np.absolute(y_test - y_test.mean())

            for j, instance in enumerate(errors[i, :]):
                # f = 1/(1+np.exp(-instance)) # função sigmoide
                # irt_matrix[i, j] = 2 - 2*f # caso utilize a função sigmoide
                f = np.clip(1/(1 + instance), 1e-4, 1-1e-4)
                irt_matrix[i, j] = f
        self.irt_matrix = pd.DataFrame(data= irt_matrix, index= names, columns = indexes).T
        if base_models == True:
            self.irt_matrix['Good'] = 0.9999 + np.random.rand(len(self.irt_matrix))*0.0001
            self.irt_matrix['Bad'] = 0.0001 + np.random.rand(len(self.irt_matrix))*0.00001
            self.irt_matrix['Medium'] = 0.5 + np.random.rand(len(self.irt_matrix))*0.0001
        return
    
def main():
    """Função principal da aplicação."""

    # Path of data set
    path_data = './data/'
    path_uci = './data/UCI - 45/'
    path_out = './beta_irt/'

    # Name of data set
    name = 'mpg'

    # Read csv
    data = pd.read_csv(path_uci + name + '.csv')
    data = data.dropna()

    X = data.iloc[:, 1:-3]
    y = data.iloc[:, 0]
    
    rd = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rd)

    # Number of instances
    len_test = len(y_test)

    cols = X_train.columns

    # Regression Models
    models = [LinearRegression(), BayesianRidge(), svm.SVR(kernel= 'linear'), svm.SVR(kernel = 'rbf'),\
         KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(),\
              AdaBoostRegressor(), MLPRegressor(), MLPRegressor(hidden_layer_sizes=(50, 50,))]

    irt = IRTModel(models= models)
    irt.fit(X_train= X_train, y_train= y_train)
    irt.irtMatrix(X_test= X_test, y_test= y_test, normalize= True, base_models= True)
    irt.irt_matrix.to_csv(path_or_buf= path_out + 'irt_data_' + name + '_s' + str(len_test) + '_f20_sd' + str(rd) +'.csv', index= False, encoding='utf-8')
    
    # X_test = pd.DataFrame(X_test, columns= cols)
    X_test['noise'] = 0
    
    print(X_test.isnull().values.any(), irt.irt_matrix.isnull().values.any())
    X_test.to_csv(path_out + 'xtest_'+ name + '_s' + str(len_test) + '_f20_sd' + str(rd) +'.csv', index= False, encoding='utf-8')
    pd.DataFrame(y_test.index, columns=['index']).to_csv('./indexes/testIndex_'+ name + '_s' + str(len_test) + '_f20_sd' + str(rd) +'.csv', index= False, encoding='utf-8')

if __name__ == "__main__":
    main()
