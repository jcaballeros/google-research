# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Modified by Jennifer Caballero <jennifer.caballero.solis@gmail.com> for compatibility with
# tensorflow for research purposes, 2023.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf

class LeakyReluPar(tf.keras.layers.Layer):
    """LeakyRelu parametric activation
    """
    def __init__(self):
        super(LeakyReluPar, self).__init__()

    def call(self, x, a):
        return (1.0 - a)/2.0 * tf.abs(x) + (1.0 + a)/2.0 * x


class LeakyReluParDeriv(tf.keras.layers.Layer):
    """Derivative of the LeakyRelu parametric activation, wrt x.
    """
    def __init__(self):
        super(LeakyReluParDeriv, self).__init__()

    def call(self, x, a):
        #return (1.0 - a)/2.0 * torch.sign(x.detach()) + (1.0 + a)/2.0
        return (1.0 - a)/2.0 * tf.sign(tf.stop_gradient(x)) + (1.0 + a)/2.0
