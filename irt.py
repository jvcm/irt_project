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

    def irtMatrix(self, X_test = [], y_test = [], normalize = False):
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
                errors[i, :] = errors[i, :]/ np.absolute(y_test - self.y_mean)

            for j, instance in enumerate(errors[i, :]):
                # f = 1/(1+np.exp(-instance)) # função sigmoide
                # irt_matrix[i, j] = 2 - 2*f # caso utilize a função sigmoide
                f = 1/(1 + instance)
                irt_matrix[i, j] = f
        self.irt_matrix = pd.DataFrame(data= irt_matrix, index= names, columns = indexes)
        return
    
def main():
    """Função principal da aplicação."""

    names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    data = pd.read_csv('data/UCI - 45/auto_mpg.csv')
    data.columns = names
    data = data.dropna()

    X = data.iloc[:, 1:-3]
    y = data.iloc[:, 0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    cols = X_train.columns
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # print(len(X_test))

    models = [LinearRegression(), BayesianRidge(), svm.SVR(kernel= 'linear'), svm.SVR(kernel = 'rbf'),\
         KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(),\
              AdaBoostRegressor(), MLPRegressor()]

    irt = IRTModel(models= models)
    irt.fit(X_train= X_train, y_train= y_train)
    irt.irtMatrix(X_test= X_test, y_test= y_test, normalize= True)

    # print(irt.irt_matrix.T)
    irt.irt_matrix.T.to_csv(path_or_buf= './beta_irt/irt_data_mpg_s79_f20_sd42.csv', index= False, encoding='utf-8')
    
    X_test = pd.DataFrame(X_test, columns= cols)
    X_test.columns = cols
    X_test['noise'] = 0
    
    print(X_test.isnull().values.any(), irt.irt_matrix.isnull().values.any())
    X_test.to_csv('./beta_irt/xtest_mpg_s79_f20_sd42.csv', index= False, encoding='utf-8')

if __name__ == "__main__":
    main()
