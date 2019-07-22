
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


def save_full_irt_parameters(errors, grades, dataset):
    [ability, difficulty, discrimination] = irt_parameters(errors)

    pd.DataFrame(data=zip(ability, grades, errors.mean(axis=1)), index=errors.index, columns=['ability', 'grade', 'mean_error']).to_csv(os.path.join('parameters', '{}-abilities.csv'.format(dataset)))
    pd.DataFrame(data=zip(difficulty, discrimination, errors.mean(axis=0)), index=errors.columns, columns=['difficulty', 'discrimination', 'mean_error']).to_csv(os.path.join('parameters', '{}-item-parameters.csv'.format(dataset)))


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

    n_iterations = 100
    filenames = ['errors-2.csv', 'errors-3.csv']

    methods = {
        'IRT': {'function': irt_mean_error, 'kwargs': {}},
        'FA': {'function': fa_mean_error, 'kwargs': {'n_components': 1}},
        'FA2': {'function': fa_mean_error, 'kwargs': {'n_components': 2}}
    }

    results = []
    for filename in filenames:
        print(filename)
        data = pd.read_csv(os.path.join('data', 'datasets', filename), header=0, index_col=0)

        errors = data[[col for col in data.columns if 'grade' not in col]]
        
        for col in errors.columns:                     
            errors[col][errors[col] == 999] = errors[col][errors[col] < 999].max() + 1

        partial = list(
            futures.map(run_experiment, [(errors, methods) for _ in tqdm(range(n_iterations), total=n_iterations)])
        )

        dataset = filename.split('.csv')[0]

        partial = pd.DataFrame(data=partial, columns=methods.keys())
        partial['dataset'] = dataset
        partial['M'] = len(errors)
        partial['N'] = len(errors.columns)

        results.append(partial)
        print(partial)
        print(partial.mean())

        save_full_irt_parameters(errors, data['grade'], dataset)

    pd.concat(results).to_csv(os.path.join('results', 'results-all.csv'), index=None)

