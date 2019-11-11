import warnings
warnings.filterwarnings('ignore')
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
import glob
import time, sys
from IPython.display import clear_output

##############################CREATING ALL REGRESSORS##############################

models = [LinearRegression(), BayesianRidge(), svm.SVR(kernel= 'linear'), svm.SVR(kernel = 'rbf', gamma= 'scale', C = 5),\
     KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(),\
          AdaBoostRegressor(), MLPRegressor(max_iter=1000, solver= 'lbfgs'), MLPRegressor(hidden_layer_sizes= (50,50), solver = 'lbfgs', max_iter=500, activation='logistic')]

names = list(map(lambda x: type(x).__name__, models))
names = names + ['Average', 'Optimal', 'Worst']

n_synth_models = 3
        
# Parameters
rd = 42

selected = './data/'
dbs = glob.glob(selected + '*.csv')

##############################READING ALL DATASETS##############################

for d, db in enumerate(dbs):
    print('\n------------------------------------------------------------\n')

    # Creating folders   
    name = db.split('/')[-1].split('.')[0]
    
    print('Data set ' + str(d + 1) + ' >>>> ' + name)
    
    if not os.path.isdir('./beta_irt/results/'+ name):
        os.system('mkdir ./beta_irt/results/'+ name)
    
    # Read file
    df = pd.read_csv(db, na_values=['?'])
    df = df.dropna()
    df = df.drop_duplicates()
    
    #Variable selection
    if df.shape[1] > 2:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Principal component analysis
#         pca = PCA(n_components= 1)
#         X_train = pca.fit_transform(X_train)
#         X_test = pca.transform(X_test)
    else: 
        X = df.iloc[:, 0].values.reshape(-1,1)
        y = df.iloc[:, -1].values
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rd)
    
    # Standard scale
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.reshape(-1,1)).reshape(1,-1)[0]
    y_test = sc_y.transform(y_test.reshape(-1,1)).reshape(1,-1)[0]
    
    print('> Training models')
    # Generate abilities/parameters for BIRT and other info.
    Irt = IRTModel(models= models)
    Irt.fit(X_train = X_train, y_train = y_train)

    # Noise
    rd = 42
    noise_std = np.linspace(0, 0.61, 21)
    max_std = noise_std.max()
    
    # Folders
    path = './beta_irt/results/'
    folder = name + '/'

    #-------------------------------------Generate-BIRT-------------------------------------#

    rep = 40
    
    noises = np.zeros((len(X_test), len(noise_std)))
    errors = np.zeros((len(noise_std), len(X_test), len(models)))
    responses = np.zeros((len(noise_std), len(X_test), len(models) + n_synth_models))
    abilities = np.zeros((len(models) + n_synth_models, len(noise_std)))
    params = np.zeros((len(noise_std), len(X_test), 2))
    
    for i, noise in enumerate(noise_std):
        
        name_ = name + '_s' + str(len(y_test)) + '_f' + str(i) + '_sd' + str(rd)
        
        for itr in range(rep):
            # Generate noise to feature in test set
#             noise_train = np.random.normal(loc=0.0, scale= noise, size= len(y_train))
            noise_test = np.random.normal(loc=0.0, scale= noise, size= len(y_test))

#             y_train_ = y_train + noise_train
            y_test_ = y_test + noise_test

            noises[:, i] += np.absolute(noise_test)

            # Generate IRT matrix
            Irt.irtMatrix(X_test= X_test, y_test= y_test_, noise_std = i, normalize= False, base_models= True, name= name, rd= rd)
            responses[i] += Irt.irt_matrix

            

            # Generate Items' parameters and Respondents' abilities
            os.chdir('./beta_irt/')
            os.system('python betairt_test.py irt_data_' + name_ + '.csv')
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
            os.system('mkdir ' + output)

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

        # X_TEST
        pd.DataFrame(data = np.hstack((X_test, y_test_.reshape(-1,1)))).to_csv(path_or_buf= output + 'test_'+ name_ + '.csv', index=False)

    #-------------------------------------Clean-Files-------------------------------------#

    os.system('rm ./beta_irt/*.csv')
    os.system('rm ' +path + folder + '*.csv')
    