# Copyright (c) 2020 Huawei Technologies Co., Ltd.
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

class DistanceMap(tf.keras.layers.Layer):
    """DistanceMap. Creates distance map (euclidian distance) from the center point. Then computes the bin values
    for each of the pixel (corresponding to the distance). For each pixel, the bin values correspond
    to a histogram of where the distance is (the bin values for each pixel must sum to 1).
    """
    def __init__(self, num_bins, bin_displacement=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.bin_displacement = bin_displacement

    def call(self, center, output_sz):
        """
        args:
            center: torch tensor with (y,x) center position
            output_sz: size of output
        output:
            bin_val: distance map tensor
        """

        center = tf.reshape(center, [-1,2])

        bin_centers = tf.reshape(tf.range(self.num_bins, dtype=tf.float32), [1, -1, 1, 1])

        k0 = tf.reshape(tf.range(output_sz[0], dtype=tf.float32), [1,1,-1,1])
        k1 = tf.reshape(tf.range(output_sz[1], dtype=tf.float32), [1,1,1,-1])

        d0 = k0 - tf.reshape(center[:, 0], [-1, 1, 1, 1])
        d1 = k1 - tf.reshape(center[:, 1], [-1, 1, 1 ,1])

        dist = tf.math.sqrt(d0*d0 + d1*d1)
        bin_diff = dist / self.bin_displacement - bin_centers

        bin_val = tf.concat([tf.nn.relu(1.0 - tf.abs(bin_diff[:, :-1, :, :])),
                    tf.clip_by_value(1.0 + bin_diff[:, -1:, :, :], 0, 1)], axis=1)


        return bin_val  # shape is (1, num_bins, output_sz[0], output_sz[1])


