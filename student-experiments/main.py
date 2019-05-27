
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import edward as ed
import six
import os
import sys
import re
import time

from hsvi.Hierarchi_klqp import Hierarchi_klqp
from models.beta_irt import Beta_IRT
import visualization.plots as vs

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.decomposition import FactorAnalysis

from edward.models import Normal,Beta,Gamma,TransformedDistribution,InverseGamma
from scipy.stats import mannwhitneyu

from tqdm import tqdm

from scoop import futures


def fa_mean_error(part1, part2, n_components=1):
    fa1 = FactorAnalysis(n_components=n_components)
    fa2 = FactorAnalysis(n_components=n_components)

    student_factors1 = fa1.fit_transform(part1) 
    question_factors1 = fa1.components_
    student_factors2 = fa2.fit_transform(part2) 
    question_factors2 = fa2.components_ 

    rec1 = np.dot(student_factors1, question_factors2)
    rec2 = np.dot(student_factors2, question_factors1)

    return (
        np.mean(
            (part1.values - rec1)**2.0
        ) + np.mean(
            (part2.values - rec2)**2.0
        )
    ) / 2.0


def irt_parameters(irt_data):
    niter = 1000

    a_prior_mean = 1.0
    a_prior_std = 1.0

    # setup Beta IRT model #

    M = irt_data.shape[1] #number of items
    C = irt_data.shape[0] #number of students


    theta = Beta(tf.ones([C]),tf.ones([C]), sample_shape=[M], name='theta')
    delta = Beta(tf.ones([M]),tf.ones([M]), sample_shape=[C], name='delta')
    a = Normal(tf.ones(M) * a_prior_mean,tf.ones([M]) * a_prior_std,
        sample_shape=[C], name='a')

    model = Beta_IRT(M,C,theta,delta,a)

    avg_error = irt_data.values.mean()

    irt_data_transformed = 1. / (1. + irt_data / avg_error)
    D = np.clip(np.float32(irt_data_transformed.values), 1e-4, 1-1e-4)


    model.init_inference(data=D,n_iter=niter)
    model.fit()

    # output ability
    ability = tf.nn.sigmoid(model.qtheta.distribution.loc).eval()

    # output difficulty and discrimination
    discrimination = model.qa.loc.eval()
    difficulty = tf.nn.sigmoid(model.qdelta.distribution.loc).eval()

    return [ability, difficulty, discrimination]


def irt_error(irt_data, ability, difficulty, discrimination):
    avg_error = irt_data.values.mean()

    M = len(difficulty)
    C = len(ability)

    #repeats through columns
    ab = np.repeat(ability.reshape(-1, 1), M, axis=1)
    #repeats through lines
    dif = np.repeat(difficulty.reshape(1, -1), C, axis=0)
    dis = np.repeat(discrimination.reshape(1, -1), C, axis=0)
    pred = 1. / (1. + (dif / (1. - dif)) ** dis * (ab / (1. - ab)) ** -dis)

    rec = avg_error * ((1. / pred) - 1.)

    return np.mean(
        (irt_data.values - rec)**2.0
    )


def irt_mean_error(part1, part2):
    [abi1, dif1, dis1] = irt_parameters(part1)
    [abi2, dif2, dis2] = irt_parameters(part2)
    return (irt_error(part1, abi1, dif2, dis2) + irt_error(part2, abi2, dif1, dis1)) / 2.0


def run_experiment(args):
    (errors, methods) = args
    n = len(errors)

    idx_permutation = np.random.choice(n, n, replace=False)

    part1 = errors.iloc[idx_permutation[:int(round(n/2))]] 
    part2 = errors.iloc[idx_permutation[int(round(n/2)):]] 
    
    partial = []
    for method in methods.keys():
        partial.append(methods[method]['function'](part1, part2, **methods[method]['kwargs']))

    return partial

if __name__ == "__main__":
    seed=1
    ed.set_seed(seed)
    np.random.seed(seed)

    filename = 'errors-3.csv'
    errors = pd.read_csv(os.path.join('data', 'datasets', filename), header=0, index_col=0)
    for col in errors.columns:                     
        errors[col][errors[col] == 999] = errors[col][errors[col] < 999].max() + 1

    n_iterations = 100

    methods = {
        'IRT': {'function': irt_mean_error, 'kwargs': {}},
        'FA': {'function': fa_mean_error, 'kwargs': {'n_components': 1}},
        'FA2': {'function': fa_mean_error, 'kwargs': {'n_components': 2}}
    }

    results = list(
        futures.map(run_experiment, [(errors, methods) for _ in tqdm(range(n_iterations), total=n_iterations)])
    )



    # for _ in tqdm(range(n_iterations)):
    #     idx_permutation = np.random.choice(n, n, replace=False)

    #     part1 = errors.iloc[idx_permutation[:int(round(n/2))]] 
    #     part2 = errors.iloc[idx_permutation[int(round(n/2)):]] 
        
    #     partial = []
    #     for method in methods.keys():
    #         partial.append(methods[method]['function'](part1, part2, **methods[method]['kwargs']))
            
    #     results.append(partial)

    results = pd.DataFrame(data=results, columns=methods.keys())

    print(results)
    print(results.mean())

    results.to_csv(os.path.join('results', 'results-3.csv'), index=None)


def print_help():
    print(usage)
    print(name_fmt)
    print('Valid data file can be generated by gen_irt_data.py')

def old_main():

    name_fmt = 'Need input data file with the name in format: irt_data_[dataset]_s[data_size]_f[noise_fraction percentile]_sd[random_seed].csv'
    usage = 'default usage: irt_data_moons_s200_f2_sd42.csv <a_prior_mean:1.> <a_prior_std:1.> <fixed_a:False>'
    seed=42
    ed.set_seed(seed)

    # parse args #
    args = sys.argv

    if len(args) < 2 or args[1] == '-h':
        print_help()
        sys.exit()

    else:
        file_name = args[1]

    # read file name #
    in_f = file_name.split('/')[-1]
    fpath = file_name[:-len(in_f)]
    in_f = re.split('_|\.',in_f)
    if len(in_f) < 7:   
        print('Wrong format of the name of data file')
        print(name_fmt)
        sys.exit()
        

    irt_data = pd.read_csv(file_name)

    dataset = in_f[2]
    result_path = './results/'+dataset

    if dataset in ['mnist','fashion']:
        niter = 2000
    else:
        niter = 1000

    dargs = {}
    for i in range(2,len(args)):
        arg = args[i].split(':')
        dargs[arg[0]] = arg[1]


    a_prior_mean = float(dargs.get('a_prior_mean', 1.))
    a_prior_std = float(dargs.get('a_prior_std',1.))
    fixed_a = dargs.get('fixed_a','False') == 'True'

    partial_name = str.join('_',in_f[2:-1])
    xtest = pd.read_csv(fpath+'xtest_'+partial_name+'.csv') # read original data

    if fixed_a:
        partial_save_name = partial_name +'_fixed_am'+str(a_prior_mean).replace('.','@')
    else:
        partial_save_name = partial_name +'_am'+str(a_prior_mean).replace('.','@')+'_as'+str(a_prior_std).replace('.','@')



    # setup Beta IRT model #

    M = irt_data.shape[0] #number of items
    C = irt_data.shape[1] #number of classifiers


    theta = Beta(tf.ones([C]),tf.ones([C]),sample_shape=[M],name='theta')
    delta = Beta(tf.ones([M]),tf.ones([M]),sample_shape=[C],name='delta')
    if fixed_a:
        a = tf.ones(M)*a_prior_mean
    else:
        a = Normal(tf.ones(M)*a_prior_mean,tf.ones([M])*a_prior_std,sample_shape=[C],name='a')

    model = Beta_IRT(M,C,theta,delta,a)

    D = np.clip(np.float32(irt_data.values), 1e-4, 1-1e-4)


    model.init_inference(data=D,n_iter=niter)
    model.fit()

    # generate output files #

    # output ability
    ability = pd.DataFrame(index=irt_data.columns)
    ability['ability'] = tf.nn.sigmoid(model.qtheta.distribution.loc).eval()
    ability.loc['stddev'] = ability.ability.std()
    ability.to_csv(result_path+'/irt_ability_vi_'+partial_save_name+'.csv') 

    # output difficulty and discrimination
    if fixed_a:
        discrimination = a.eval()
    else:   
        discrimination = model.qa.loc.eval()

    difficulty = tf.nn.sigmoid(model.qdelta.distribution.loc).eval()
    if not dataset in ['fashion','mnist']:
        if not fixed_a:
            fig = vs.plot_parameters(xtest.values[:,:-1], difficulty, discrimination)
            fig.savefig(result_path+'/irt_parameters_vi_'+partial_save_name+'.pdf') 

    parameters = pd.DataFrame(index=irt_data.index)
    parameters['difficulty'] = difficulty
    parameters['discrimination'] = discrimination
    parameters.to_csv(result_path+'/irt_parameters_vi_'+partial_save_name+'.csv',index=False)

    # visualize correlation between difficulty and response
    irt_prob_avg = irt_data.mean(axis=1)
    if fixed_a:
        fig = vs.plot_item_parameters_corr(irt_prob_avg,difficulty,xtest.noise)
        
    else:
        fig = vs.plot_item_parameters_corr(irt_prob_avg,difficulty,xtest.noise,discrimination)

    fig.savefig(result_path+'/irt_itemparam_corr_'+partial_save_name+'.pdf')

    # output performance of detected noisy points
    if not fixed_a:
        if not dataset in ['fashion','mnist']:
            fig = vs.plot_noisy_points(xtest,discrimination)
            fig.savefig(result_path+'/dnoise_visual_'+partial_save_name+'.pdf')
        #print(xtest.loc[xtest.noise>0].index)
        correct_noise_sum = xtest.loc[discrimination<0,'noise'].sum()
        true_noise_sum = xtest['noise'].sum()
        predict_noise_sum = (discrimination<0).sum()
        if predict_noise_sum < 1:
            print('None noise is found!')
            precision = 0.
        else:   
            precision = 1.*correct_noise_sum/predict_noise_sum

        if true_noise_sum < 1:
            print('None noise is injected!')
            recall = 0.
        else:   
            recall = 1.*correct_noise_sum/true_noise_sum
        print('precision', precision, 'recall',recall)
        with open(result_path+'/dnoise_performance_'+partial_save_name+'.txt', 'w') as pfile:
            pfile.write('precision = '+str(precision)+'\n'+'recall = '+str(recall))








