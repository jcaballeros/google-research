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

class LocalGOCor(Model):
  """Model to improve the local correlation layer output based on Truong et al. 2020 GOCor paper"""
  def __init__(self,
               use_bfloat16=False):
    super(LocalGOCor, self).__init__()
    self._use_bfloat16 = use_bfloat16
    if use_bfloat16:
      self._dtype_policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    else:
      self._dtype_policy = tf.keras.mixed_precision.Policy('float32')

  def call(self, reference_feature, query_feature):
    # Just return the reference feature for now
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
