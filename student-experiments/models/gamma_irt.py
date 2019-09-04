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

from edward.models import Normal,Beta,Gamma,TransformedDistribution,InverseGamma

ds = tf.contrib.distributions

class Gamma_IRT:

    def __init__(self, M, C, theta_prior, delta_prior, a_prior, n_iter=1000, n_print=100):

        self.M = M
        self.C = C
        self.theta_prior = theta_prior # prior of ability
        self.delta_prior = delta_prior # prior of difficulty
        self.a_prior = a_prior  # prior of discrimination
        self.n_iter = n_iter
        self.n_print = n_print

        if isinstance(a_prior,ed.RandomVariable):
            # variational posterior of discrimination
            self.qa = Normal(loc=tf.Variable(tf.random_normal([M])), scale=tf.nn.softplus(tf.Variable(tf.ones([M])*.5)),name='qa')
        else:
            self.qa = a_prior

        with tf.variable_scope('local'):
            # variational posterior of ability
            self.qtheta = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([C])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([C])))),\
                                                           bijector=ds.bijectors.Sigmoid(), sample_shape=[M],name='qtheta')
            # variational posterior of difficulty
            self.qdelta = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([M])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([M])))), \
                                                            bijector=ds.bijectors.Sigmoid(), sample_shape=[C],name='qdelta')

        alpha = (self.qdelta/tf.transpose(self.qtheta))**self.qa

        beta = ((1. - self.qdelta)/(1. - tf.transpose(self.qtheta)))**self.qa

        # observed variable
        self.x = Gamma(tf.transpose(alpha),tf.transpose(beta))        

    
    def fit(self, data, local_iter=50):
        ability, difficulty, discrimination = None, None, None

        sess = tf.InteractiveSession()
        with sess.as_default():
            self.inference = Hierarchi_klqp(latent_vars={self.a_prior:self.qa}, data={self.x:data}, \
                            local_vars={self.theta_prior:self.qtheta,self.delta_prior:self.qdelta},local_data={self.x:data})
            
            self.inference.initialize(auto_transform=False,n_iter=self.n_iter,n_print=self.n_print)

            tf.global_variables_initializer().run()

            for jj in range(self.inference.n_iter):  
                if isinstance(self.a_prior,ed.RandomVariable):
                    for _ in range(local_iter):
                        self.inference.update(scope='local')
                info_dict = self.inference.update(scope='global')
                self.inference.print_progress(info_dict)

            ability = tf.nn.sigmoid(self.qtheta.distribution.loc).eval()
            discrimination = self.qa.loc.eval()
            difficulty = tf.nn.sigmoid(self.qdelta.distribution.loc).eval()

        return ability, difficulty, discrimination
