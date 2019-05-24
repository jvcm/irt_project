import os
import pandas as pd
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from irt import IRTModel
from sklearn import svm
from sklearn.linear_model import SGDRegressor, LinearRegression, BayesianRidge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from beta_irt.visualization.plots import newline
from beta_irt.visualization.plots import plot_parameters
from irt import beta_irt
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import gridspec
from sklearn.preprocessing import StandardScaler
import edward as ed
import os

#-------------------------------------Pre-processing-------------------------------------#

# Path
path_data = './data/'
path_uci = './data/UCI - 45/'

# Name of data set
name = 'mpg'

# Read csv
data = pd.read_csv(path_uci + name + '.csv')
data = data.dropna()

# Parameters
rd = 42
noise_std = np.linspace(0, 1.6, 20)
max_std = noise_std.max()

# Variable selection
X = data.iloc[:, 1:-3].values
y = data.iloc[:, 0]

# Split data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rd)
indexes = list(y_train.index)

# Principal component analysis
pca = PCA(n_components= 1)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Standard scale
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Regression Models
models = [LinearRegression(), BayesianRidge(), svm.SVR(kernel= 'linear'), svm.SVR(kernel = 'rbf', gamma= 'scale', C = 5),\
     KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(),\
          AdaBoostRegressor(), MLPRegressor(max_iter=1000, solver= 'lbfgs'), MLPRegressor(hidden_layer_sizes= (50,50), solver = 'lbfgs', max_iter=500, activation='logistic')]

# Generate abilities/parameters for BIRT and other info.
Irt = IRTModel(models= models)
Irt.fit(X_train = X_train, y_train = y_train)

# Edward - set seed
ed.set_seed(rd)

# Folders
path = './beta_irt/results/'
folder = name + '/'

#-------------------------------------Generate-BIRT-------------------------------------#

noises = np.zeros((len(X_test), len(noise_std)))
errors = np.zeros((len(noise_std), len(X_test), len(models)))
responses = np.zeros((len(noise_std), len(X_test), len(models) + 3))
abilities = np.zeros((len(models) + 3, len(noise_std)))
params = np.zeros((len(noise_std), len(X_test), 2))

rep = 5
for i, noise in enumerate(noise_std):
    for itr in range(rep):
        # Generate noise to feature in test set
        noise_test = np.random.normal(loc=0.0, scale= noise, size= len(X_test))
        noises[:, i] += np.absolute(noise_test)
        X_test_ = X_test + noise_test.reshape(-1,1)

        # Generate IRT matrix
        Irt.irtMatrix(X_test= X_test_, y_test= y_test, noise_std = i, normalize= True, base_models= True, name= name, rd= rd)
        responses[i] += Irt.irt_matrix

        name_ = name + '_s' + str(len(y_test)) + '_f' + str(i) + '_sd' + str(rd)

        # Generate Items' parameters and Respondents' abilities
        os.chdir('./beta_irt/')
        os.system('run -i betairt_test.py irt_data_' + name_ + '.csv')
        os.chdir('..')

        errors[i] += pd.read_csv('./beta_irt/errors_' + name_ + '.csv').iloc[:, :].values
        abilities[:, i] += pd.read_csv(path + folder + 'irt_ability_vi_'+ name_ +'_am1@0_as1@0.csv').iloc[:-1, 1:].values.reshape(1,-1)[0]
        params[i] += pd.read_csv(path + folder + 'irt_parameters_vi_'+ name_ +'_am1@0_as1@0.csv').values
    
    responses[i] /= rep
    noises[:, i] /= rep
    errors[i] /= rep
    abilities[:, i] /= rep
    params[i] /= rep
    
    # Move files to folder    
    output = './Results_IRT/'+ folder + 'noise_' + str(i) + '/'
    if not os.path.isdir('./Results_IRT/'+ folder):
        os.system('mkdir ./Results_IRT/'+ folder)
    if not os.path.isdir(output):
        os.system('mkdir' + output)
        
    # ABILITY
    pd.DataFrame(data= np.hstack((pd.read_csv(path + folder + 'irt_ability_vi_'+ name_ +'_am1@0_as1@0.csv').iloc[:-1, 0].values.reshape(-1,1),
                              abilities[:, i].reshape(-1,1))),
             columns = ['Models','Ability']).to_csv(path_or_buf= output + 'irt_ability_vi_'+ name_ + '.csv', index=False)
    
    # PARAMETERS
    pd.DataFrame(data= params[i], columns=['Difficulty','Discrimination']).to_csv(path_or_buf= output + 'irt_parameters_vi_'+ name_ + '.csv', index=False)
    
    # NOISE
    pd.DataFrame(data= noises[:, i].reshape(-1,1), columns=['Noise']).to_csv(path_or_buf= output + 'noise_'+ name_ + '.csv', index=False)
    
    # ERRORS
    pd.DataFrame(data= errors[i], columns=pd.read_csv(path + folder + 'irt_ability_vi_'+ name_ +'_am1@0_as1@0.csv').iloc[:-4, 0].values).to_csv(path_or_buf= output + 'errors_'+ name_ + '.csv', index=False)
    
    # RESPONSES
    pd.DataFrame(data= responses[i], columns=pd.read_csv(path + folder + 'irt_ability_vi_'+ name_ +'_am1@0_as1@0.csv').iloc[:-1, 0].values).to_csv(output + 'irt_data_' + name_ + '.csv', index=False)

#-------------------------------------Clean-Files-------------------------------------#

os.system('./beta_irt/*.csv')
os.system(path + folder + '*.csv')
