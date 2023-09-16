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

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU

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

class LocalGOCor(Model):
  """Model to improve the local correlation layer output based on Truong et al. 2020 GOCor paper"""
  def __init__(self,
               use_bfloat16=False,
               num_iter=3):
    super(LocalGOCor, self).__init__()
    self._use_bfloat16 = use_bfloat16
    if use_bfloat16:
      self._dtype_policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    else:
      self._dtype_policy = tf.keras.mixed_precision.Policy('float32')

    self._num_iter = num_iter
    self._initializer = LocalGOCorInitializer(use_bfloat16=use_bfloat16)
    self._score_activation = LeakyReLU(dtype=self._dtype_policy)


  def call(self, reference_feature, query_feature):
    w = self._initializer(reference_feature)

    w = uflow_utils.compute_local_cost_volume(w, reference_feature, 4)
    w_act = self._score_activation(w)
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
