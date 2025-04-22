# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of MED (Mutual information based Entropy
Disentanglement) and Top-k MED.

Disentanglement codes borrowed from dci.
Mutual information calculation code borrowed from mig.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy
from six.moves import range
from sklearn import ensemble, linear_model, preprocessing
import sklearn as sk


def med(importance_matrix, topk=-1):
  """Computes score based on both training and testing codes and factors."""

  if topk > 0: # top-k D and C
    pick_index = pick_by_dis_per_factor(importance_matrix, topk)
    importance_matrix = importance_matrix[pick_index, :]
    return disentanglement(importance_matrix), completeness(importance_matrix)
  else:
      return disentanglement(importance_matrix), completeness(importance_matrix)


def _histogram_discretize(target, num_bins=20):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized

def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sk.metrics.mutual_info_score(ys[j, :], mus[i, :])
  return m

def compute_importance_mi(x, y, train_perc=0.75, dyn_offset=0):
  """Compute importance matrix based on Mutual Information and informativeness based on Logistic."""
  num_factors = y.shape[0]
  num_points = y.shape[1]
  num_codes = x.shape[0]
  x_train = x[:, :int(num_points * train_perc)]
  x_test = x[:, int(num_points * train_perc):]
  y_train = y[:, :int(num_points * train_perc)]
  y_test = y[:, int(num_points * train_perc):]
  # Caculate importance by MI like MIG.
  discretized_mus = _histogram_discretize(x_train)
  m = discrete_mutual_info(discretized_mus, y_train)
  # m's shape is num_codes x num_factors
  # Norm by factor sum.
  importance_matrix = np.divide(m, m.sum(axis=0))

  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = linear_model.LogisticRegression()
    # Some case fails to converge, add preprocessing to scale data to zero mean
    # and unit std.
    scaler = preprocessing.StandardScaler().fit(x_train.T)
    x_train_scale = scaler.transform(x_train.T)
    x_test_scale = scaler.transform(x_test.T)
    model.fit(x_train_scale, y_train[i, :])
    #NOTE: Copy and paste from disentangle_lib. It's acctually train acc here
    train_loss.append(np.mean(model.predict(x_train_scale) == np.array(y_train[i, :])))
    test_loss.append(np.mean(model.predict(x_test_scale) == np.array(y_test[i, :])))
  return importance_matrix, np.mean(test_loss), np.mean(test_loss[dyn_offset:])

def pick_by_dis_per_factor(importance_matrix, k):
    """ Selection process of Top-k MED. For each factor, selects the most
    k disentangled dimensions. """
    latent_num, factor_num= importance_matrix.shape
    dis_per_code = disentanglement_per_code(importance_matrix)
    sort_index = np.argsort(-1 * dis_per_code)
    factor_per_code = np.argmax(importance_matrix, axis=1)
    factor_dim = [[] for _ in range(factor_num)]
    is_full = [False for _ in range(factor_num)]
    for dim in sort_index:
        cur_factor = factor_per_code[dim]
        if len(factor_dim[cur_factor]) < k:
            factor_dim[cur_factor].append(dim)
        else:
            is_full[cur_factor] = True
        if all(is_full) == True:
            break
    select_index = []
    for fac_d in factor_dim:
        select_index.extend(fac_d)
    return list(set(select_index))

def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)