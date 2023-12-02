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

class LeakyReluPar(Model):
  def __init__(self):
    super(LeakyReluPar, self).__init__()

  def call(self, x, alpha):
    return (1.0 - alpha)/2.0 * tf.math.abs(x) + (1.0 + alpha)/2.0 * x

class LeakyReluParDeriv(Model):
  def __init__(self):
    super(LeakyReluParDeriv, self).__init__()

  def call(self, x, alpha):
    return (1.0 - alpha)/2.0 * tf.math.sign(tf.stop_gradient(x)) + (1.0 + alpha)/2.0

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
               init_gauss_sigma=1.0,
               channels=32,
               min_filter_reg=1e-5,
               init_step_length=1.0,
               init_filter_reg=1e-2):
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
    self._target_mask_predictor = tf.keras.Sequential()
    self._target_mask_predictor.add(Conv2D(filters=1, kernel_size=1, use_bias=False,
        kernel_initializer=target_mask_weights_initializer, activation='sigmoid'))

    self._score_activation = LeakyReluPar()
    self._score_activation_deriv = LeakyReluParDeriv()

    # Convolution for dimensionality reduction
    self._chan_reduction_mr = Conv2D(filters=channels, kernel_size=1, strides=1, padding='valid')
    self._chan_reduction_lr = Conv2D(filters=channels, kernel_size=1, strides=1, padding='valid')

    # Regularization
    self._w_reg = tf.Variable(initial_value=(init_filter_reg * tf.ones(1)), trainable=True)
    self._min_filter_reg = min_filter_reg
    self._log_step_length = tf.Variable(tf.math.log(init_step_length) * tf.ones(1), trainable=True)

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
    # Get initial value for filter map w
    w = self._initializer(reference_feature)

    # Initialize filter map w regularization weights
    _, width, height, ref_feature_chan = reference_feature.shape.as_list()
    reg_weight = tf.clip_by_value(self._w_reg * self._w_reg,
        clip_value_min=(self._min_filter_reg**2)/(ref_feature_chan**2), clip_value_max=tf.float32.max)
    step_length = tf.math.exp(self._log_step_length)
    correlation_channels = (2 * self._max_displacement + 1)**2
    v_plus = tf.ones([width, height, correlation_channels])
    v_minus = tf.zeros_like([width, height, correlation_channels])
    distance_map = self.compute_distance_map()
    y = tf.reshape(self._target_map(distance_map), [1, 1, 1, -1])
    v_plus = tf.reshape(self._spatial_weight_predictor(distance_map), [1, 1, 1, -1])
    weight_m = tf.reshape(self._target_mask_predictor(distance_map), [1, 1, 1, -1])

    for i in range(self._num_iter):
      # Compute the correlation between filter_map and the reference features and initialize v+ and v-
      c_fref_w = uflow_utils.compute_local_cost_volume(w, reference_feature, self._max_displacement)

      # Compute sigma and its derivative
      act_scores_filter_w_ref = v_plus * self._score_activation(c_fref_w, weight_m)
      grad_act_scores_by_filter = v_plus * self._score_activation_deriv(c_fref_w, weight_m)

      # Compute L_r
      loss_ref_residuals = act_scores_filter_w_ref - v_plus * y
      mapped_residuals = grad_act_scores_by_filter * loss_ref_residuals

      # Reduce the number of channels of mapped_residuals for the cost voolume computation:
      mapped_residuals = self._chan_reduction_mr(mapped_residuals)

      # Compute gradient of L_r
      filter_grad_loss_ref = uflow_utils.compute_local_cost_volume(mapped_residuals, reference_feature, self._max_displacement)
      filter_grad_loss_ref = self._chan_reduction_lr(filter_grad_loss_ref)

      # Update filter map w
      filter_grad_reg = reg_weight * w
      filter_grad = filter_grad_reg + filter_grad_loss_ref
      scores_filter_grad_w_ref = uflow_utils.compute_local_cost_volume(filter_grad, reference_feature, self._max_displacement)
      scores_filter_grad_w_ref = grad_act_scores_by_filter * scores_filter_grad_w_ref
      alpha_den = tf.math.reduce_sum(scores_filter_grad_w_ref * scores_filter_grad_w_ref, axis=-1, keepdims=True)
      alpha_num = tf.math.reduce_sum(filter_grad * filter_grad, axis=-1, keepdims=True)
      alpha = alpha_num / alpha_den
      w = w + (alpha * step_length) * filter_grad

    return w

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
