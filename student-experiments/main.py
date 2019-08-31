
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import datetime
import numpy as np
import tensorflow as tf
import edward as ed
import six
import os
import sys
import re
import time
import warnings
import ntpath

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

    question_factors1 = fa1.fit_transform(part1) 
    student_factors1 = fa1.components_
    question_factors2 = fa2.fit_transform(part2) 
    student_factors2 = fa2.components_ 

    rec1 = np.dot(question_factors2, student_factors1)
    rec2 = np.dot(question_factors1, student_factors2)
    
    error = (
        np.mean(
            (part1.values - rec1)**2.0
        ) + np.mean(
            (part2.values - rec2)**2.0
        )
    ) / 2.0

    return error

def irt_parameters(irt_data, n_iterations=100):
    niter = n_iterations

    a_prior_mean = 1.0
    a_prior_std = 1.0

    # setup Beta IRT model #

    M = irt_data.shape[0] #number of items
    C = irt_data.shape[1] #number of students

    theta = Beta(tf.ones([C]),tf.ones([C]), sample_shape=[M], name='theta')
    delta = Beta(tf.ones([M]),tf.ones([M]), sample_shape=[C], name='delta')
    a = Normal(tf.ones(M) * a_prior_mean,tf.ones([M]) * a_prior_std,
        sample_shape=[C], name='a')

    model = Beta_IRT(M,C,theta,delta,a)

    irt_data_transformed = 1. / (1. + irt_data)
    D = np.clip(np.float32(irt_data_transformed.values), 1e-4, 1-1e-4)


    model.init_inference(data=D,n_iter=niter)
    model.fit()

    # output ability
    ability = tf.nn.sigmoid(model.qtheta.distribution.loc).eval()

    # output difficulty and discrimination
    discrimination = model.qa.loc.eval()
    difficulty = tf.nn.sigmoid(model.qdelta.distribution.loc).eval()

    model.close()

    return [ability, difficulty, discrimination]


def irt_error(irt_data, ability, difficulty, discrimination):

    M = len(difficulty)
    C = len(ability)

    #repeats through columns
    ab = np.repeat(ability.reshape(1, -1), M, axis=0)
    #repeats through lines
    dif = np.repeat(difficulty.reshape(-1, 1), C, axis=1)
    dis = np.repeat(discrimination.reshape(-1, 1), C, axis=1)
    pred = 1. / (1. + (dif / (1. - dif)) ** dis * (ab / (1. - ab)) ** -dis)
    
    rec = (1. / pred) - 1.

    return np.mean(
        (irt_data.values - rec)**2.0
    )


def irt_mean_error(part1, part2, n_iterations=100):
    [abi1, dif1, dis1] = irt_parameters(part1, n_iterations=n_iterations)
    [abi2, dif2, dis2] = irt_parameters(part2, n_iterations=n_iterations)
    error = (irt_error(part1, abi1, dif2, dis2) + irt_error(part2, abi2, dif1, dis1)) / 2.0
    abi1, dif1, dis1 = None, None, None
    abi2, dif2, dis2 = None, None, None
    return error


def save_full_irt_parameters(errors, grades, dataset):
    [ability, difficulty, discrimination] = irt_parameters(errors)

    pd.DataFrame(data=zip(ability, grades, errors.mean(axis=1)), index=errors.index, columns=['ability', 'grade', 'mean_error']).to_csv(os.path.join('parameters', '{}-abilities.csv'.format(dataset)))
    pd.DataFrame(data=zip(difficulty, discrimination, errors.mean(axis=0)), index=errors.columns, columns=['difficulty', 'discrimination', 'mean_error']).to_csv(os.path.join('parameters', '{}-item-parameters.csv'.format(dataset)))


def run_experiment(args):
    (errors, methods) = args
    n = errors.shape[1]

    idx_permutation = np.random.choice(n, n, replace=False)
    
    part1 = errors.iloc[:, idx_permutation[:int(round(n/2))]] 
    part2 = errors.iloc[:, idx_permutation[int(round(n/2)):]] 
    
    partial = []
    for method in methods.keys():
        partial.append(methods[method]['function'](part1, part2, **methods[method]['kwargs']))

    part1 = None
    part2 = None
    
    return partial


def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs all the experiments
                                     with the given arguments''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', dest='seed_num', type=int,
                        default=42,
                        help='Seed for the random number generator')
    parser.add_argument('-m', '--mciterations', dest='mc_iterations', type=int,
                        default=50,
                        help='Number of Markov Chain iterations')
    parser.add_argument('-i', '--irtiterations', dest='irt_iterations', type=int,
                        default=100,
                        help='Number of Beta-IRT iterations')
    parser.add_argument('-d', '--dataset', dest='dataset',
                        type=str,
                        default='./data/datasets/normalised/stats101-1.csv',
                        help='''Dataset path''')
    return parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    ed.set_seed(args.seed_num)
    np.random.seed(args.seed_num)

    n_iterations = args.mc_iterations

    methods = {
        'IRT': {'function': irt_mean_error, 'kwargs': {'n_iterations': args.irt_iterations}},
        'FA': {'function': fa_mean_error, 'kwargs': {'n_components': 1}},
        'FA2': {'function': fa_mean_error, 'kwargs': {'n_components': 2}}
    }

    dataset_path = args.dataset
    print(dataset_path)
    
    errors = pd.read_csv(dataset_path, header=0)

    partial = list(
        futures.map(run_experiment, [(errors, methods) for _ in tqdm(range(n_iterations), total=n_iterations)])
    )

    dataset = ntpath.basename(dataset_path).split('.csv')[0]

    partial = pd.DataFrame(data=partial, columns=methods.keys())
    partial['dataset'] = dataset
    partial['M'] = len(errors)
    partial['N'] = len(errors.columns)

    print(partial.mean())

    partial.to_csv(os.path.join('results', 'results-{}.csv'.format(dataset)), index=None)
        # save_full_irt_parameters(errors, data['grade'], dataset)

