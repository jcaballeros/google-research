# Copyright 2023 Jennifer Caballero <jennifer.caballero.solis@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D

from uflow import uflow_utils

class LocalGOCorInitializer(Model):
  def __init__(self,
               use_bfloat16=False):
    super(LocalGOCorInitializer, self).__init__()
    self._use_bfloat16 = use_bfloat16
    if use_bfloat16:
      self._dtype_policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    else:
      self._dtype_policy = tf.keras.mixed_precision.Policy('float32')

    self._beta = tf.Variable(tf.ones(1, tf.float32))

  def call(self, feat):
    w0 = self._beta * (feat / (tf.reduce_mean(feat*feat, axis=-1, keepdims=True) + 1e-6))
    return w0

def target_map_weights_initializer(shape, dtype=None):
  num_bins = 10
  bin_displacement = 0.5
  init_gauss_sigma = 1.0
  d = tf.reshape(tf.range(num_bins, dtype=tf.float32),
                 shape) * bin_displacement
  init_gauss = tf.math.exp(-1/2 * (d / init_gauss_sigma)**2)
  init_gauss_weights = init_gauss - tf.math.reduce_min(init_gauss)
  return init_gauss_weights

def target_mask_weights_initializer(shape, dtype=None):
  num_bins = 10
  v_minus_init_factor = 4.0
  bin_displacement = 0.5
  d = tf.reshape(tf.range(num_bins, dtype=tf.float32), shape) * bin_displacement
  weights = v_minus_init_factor * tf.math.tanh(2.0 - d)
  return weights

class LocalGOCor(Model):
  """Model to improve the local correlation layer output based on Truong et al. 2020 GOCor paper"""
  def __init__(self,
               use_bfloat16=False,
               num_iter=3,
               num_bins=10,
               max_displacement=4,
               bin_displacement=0.5,
               init_gauss_sigma=1.0):
    super(LocalGOCor, self).__init__()
    self._use_bfloat16 = use_bfloat16
    if use_bfloat16:
      self._dtype_policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    else:
      self._dtype_policy = tf.keras.mixed_precision.Policy('float32')

    self._num_iter = num_iter
    self._initializer = LocalGOCorInitializer(use_bfloat16=use_bfloat16)
    self._num_bins = num_bins
    self._max_displacement = max_displacement
    self._bin_displacement = bin_displacement

    # Initialize ideal correlation
    self._target_map = Conv2D(filters=1, kernel_size=1, use_bias=False,
        kernel_initializer=target_map_weights_initializer)
    self._spatial_weight_predictor = Conv2D(filters=1, kernel_size=1, use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(1.0))
    self._target_mask_predictor = Conv2D(filters=1, kernel_size=1, use_bias=False,
              kernel_initializer=target_mask_weights_initializer, activation='sigmoid')

  def sigma_smooth(self, c, v_plus, v_minus):
    ((v_plus-v_minus)/2.0)*tf.abs(c) + ((v_plus+v_minus)/2.0)*c

  def sigma_smooth_deriv(self, c, v_plus, v_minus):
    ((v_plus-v_minus)/2.0)*tf.math.sign(c) + ((v_plus+v_minus)/2.0)

  def compute_distance_map(self):
    search_size = 2 * self._max_displacement + 1
    center = tf.constant([self._max_displacement, self._max_displacement])
    histogram_bins = tf.reshape(tf.range(self._num_bins, dtype=tf.float32),
        [1, 1, 1, self._num_bins])
    displacement_range = tf.range(-1*self._max_displacement, self._max_displacement+1,
        dtype=tf.float32)
    k0 = tf.reshape(displacement_range, [1, search_size, 1, 1])
    k1 = tf.reshape(displacement_range, [1, 1, search_size, 1])
    dist = tf.math.sqrt(k0*k0 + k1*k1)
    bin_diff = dist / self._bin_displacement - histogram_bins
    bin_val = tf.concat([tf.nn.relu(1.0 - tf.abs(bin_diff[:, :-1, :, :])),
        tf.clip_by_value(1.0 + bin_diff[:, -1:, :, :], 0, 1)], axis=1)
    return bin_val

  def call(self, reference_feature, query_feature):
    w = self._initializer(reference_feature)

    c_fref_w = uflow_utils.compute_local_cost_volume(w, reference_feature, self._max_displacement)
    v_plus = tf.ones_like(c_fref_w)
    v_minus = tf.zeros_like(c_fref_w)
    sigma_n = self.sigma_smooth(c_fref_w, v_plus, v_minus)
    sigma_n_deriv = self.sigma_smooth_deriv(c_fref_w, v_plus, v_minus)
    distance_map = self.compute_distance_map()
    target_map = self._target_map(distance_map)
    v_plus = self._spatial_weight_predictor(distance_map)
    weight_m = self._target_mask_predictor(distance_map)

    return reference_feature

class GlobalGOCor(Model):
  """Model to improve the global correlation layer output based on Truong et al. 2020 GOCor paper"""
  def __init__(self,
               use_bfloat16=False):
    super(GlobalGOCor, self).__init__()
    self._use_bfloat16 = use_bfloat16
    if use_bfloat16:
      self._dtype_policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    else:
      self._dtype_policy = tf.keras.mixed_precision.Policy('float32')

  def call(self, reference_feature, query_feature):
    # Just return the reference feature for now
    return reference_feature
