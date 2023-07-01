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

import tensorflow as tf
import math
from uflow.distance import DistanceMap
from . import activation
from . import correlation
#from . import uflow_model

class LocalCorrSimpleInitializer(tf.keras.layers.Layer):
    """Local GOCor initializer module. 
    Initializes the Local GOCor filter through a simple norm operation
    args:
        filter_size: spatial kernel size of filter
    """

    def __init__(self, filter_size=1):
        super(LocalCorrSimpleInitializer, self).__init__()
        assert filter_size == 1

        self.filter_size = filter_size
        self.scaling = tf.Variable(tf.ones(1))

    def call(self, feat):
        """Initialize filter.
        args:
            feat: input features (sequences, H, W, feat_dim)
        output:
            weights: initial filters (sequences, H, W, feat_dim)
        """

        weights = feat / (tf.reduce_mean(feat*feat, axis=-1, keepdims=True) +
                          1e-6)
        weights = self.scaling * weights
        return weights

class LocalGOCorrOpt(tf.keras.layers.Layer):
    """Local GOCor optimizer module. 
    Optimizes the LocalGOCor filter map on the reference image.
    args:
        num_iter: number of iteration recursions to run in the optimizer
        init_step_length: initial step length factor
        init_filter_reg: initialization of the filter regularization parameter
        target_sigma: standard deviation for the correlation volume label in the reference image
        test_loss: Loss to use for the test data
        min_filter_reg: an epsilon thing to avoid devide by zero
    """
    def __init__(self, num_iter=3, init_step_length=1.0, init_filter_reg=1e-2,
                 min_filter_reg=1e-5, num_dist_bins=10, bin_displacement=0.5, init_gauss_sigma=1.0,
                 v_minus_act='sigmoid', v_minus_init_factor=4.0, search_size=9,
                 apply_query_loss=False, reg_kernel_size=3, reg_inter_dim=1, reg_output_dim=1):
        super(LocalGOCorrOpt, self).__init__()
        self.apply_query_loss = False
        assert search_size == 9
        self.num_iter = num_iter
        self.min_filter_reg = min_filter_reg
        self.search_size = search_size

        #self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.log_step_length = tf.Variable(math.log(init_step_length) * tf.ones(1))
        #self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.filter_reg = tf.Variable(init_filter_reg * tf.ones(1))

        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        
        # for the reference loss L_r
        # Distance coordinates
        d = tf.reshape(tf.range(num_dist_bins, dtype=tf.float32), [1,-1,1,1]) * bin_displacement
        
        # initialize the label map predictor y'_theta
        if init_gauss_sigma == 0:
            init_gauss = tf.zeros_like(d)
            init_gauss[0, 0, 0, 0] = 1
        else:
            init_gauss = tf.math.exp(-1/2 * (d / init_gauss_sigma)**2)
        self.init_gauss = init_gauss
        #self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        # initializer = tf.keras.initializers.Ones()
        # values = initializer(shape=(d.shape))
        self.label_map_predictor = tf.keras.layers.Conv2D(1, input_shape=(num_dist_bins, 1), kernel_size=1, use_bias=False)

        #self.label_map_predictor.weight.data = init_gauss - tf.reduce_min(init_gauss)
        
        # initialize the weight v_plus predictor, here called spatial_weight_predictor
        #self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        #self.spatial_weight_predictor = tf.keras.layers.Conv2D(1, input_shape=(num_dist_bins, 1), kernel_size=1, use_bias=False)
        self.spatial_weight_predictor = tf.keras.layers.Conv2D(1, input_shape=(num_dist_bins, 1), kernel_size=1, use_bias=False)
        #self.spatial_weight_predictor.weight.data.fill_(1.0)
        
        # initialize the weights m predictor m_theta, here called target_mask_predictor
        # the weights m at then used to compute the weights v_minus, as v_minus = m * v_plus
        self.num_bins = num_dist_bins
        print('numb_dist_bins is: ' + str(num_dist_bins)) 

        # init_v_minus = [tf.keras.layers.Conv2D(1, input_shape=(4,num_dist_bins, 9,9), kernel_size=1, use_bias=False)]
        # init_w = v_minus_init_factor * tf.math.tanh(2.0 - d)
        # self.v_minus_act = v_minus_act
        # print("before")
        # self.target_mask_predictor = tf.keras.Sequential(*init_v_minus)
        # print("after")

        # if v_minus_act == 'sigmoid':
        #     #init_v_minus.append(nn.Sigmoid())
        #     self.target_mask_predictor.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))
        # elif v_minus_act == 'linear':
        #     #init_w = torch.sigmoid(init_w)
        #     init_w = tf.math.sigmoid(init_w)
        # else:
        #     raise ValueError('Unknown activation')

        init_v_minus = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, kernel_size=1, use_bias=False, padding='same', strides=(1, 1), input_shape=(9, 9, 10))
        ])

        # Calculate `init_w` using TensorFlow operations
        init_w = v_minus_init_factor * tf.math.tanh(2.0 - d)

        # Set the activation function for `init_v_minus` based on `v_minus_act`
        v_minus_act = 'sigmoid'  # Replace this with the actual value
        if v_minus_act == 'sigmoid':
            init_v_minus.add(tf.keras.layers.Activation('sigmoid'))
        elif v_minus_act == 'linear':
            init_w = tf.math.sigmoid(init_w)
        else:
            raise ValueError('Unknown activation')

        # Assuming you have defined the `target_mask_predictor` model as Sequential
        # Add the `init_v_minus` layers to `target_mask_predictor`
        self.target_mask_predictor = tf.keras.Sequential()
        for layer in init_v_minus.layers:
            self.target_mask_predictor.add(layer)

        #self.target_mask_predictor = tf.keras.Sequential()
        #self.target_mask_predictor[0].weight.data = init_w
        #self.init_target_mask_predictor = init_w.clone()  # for plotting
        self.init_target_mask_predictor = tf.identity(init_w)  # for plotting
        
        # initialize activation function sigma (to apply to the correlation score between the filter map and the ref)
        self.score_activation = activation.LeakyReluPar()
        self.score_activation_deriv = activation.LeakyReluParDeriv()

    def _plot_weights(self, save_dir):
        plot_local_gocor_weights(save_dir, self.init_gauss, self.label_map_predictor, self.init_target_mask_predictor,
                                 self.target_mask_predictor, self.v_minus_act, self.num_bins,
                                 self.spatial_weight_predictor)

    def call(self, filter_map, reference_feat, query_feat=None, num_iter=None, compute_losses=False):
        """
        Apply optimization loop on the initialized filter map
        args:
            filter_map: initial filters, shape is (b, feat_dim, H, W)
            reference_feat: features from the reference image, shape is (b, feat_dim, H, W)
            query_feat: features from the query image, shape is (b, feat_dim, H, W)
            num_iter: number of iteration, to overwrite num_iter given in init parameters
            compute_losses: compute intermediate losses
        output:
            filters and losses
        """
        if num_iter is None:
            num_iter = self.num_iter

        num_sequences = reference_feat.shape[0]
        num_filters = reference_feat.shape[-2] * reference_feat.shape[-1]
        feat_sz = (reference_feat.shape[-2], reference_feat.shape[-1])
        feat_dim = reference_feat.shape[-3]

        # Compute distance map
        dist_map_sz = (self.search_size, self.search_size)
        center = tf.constant([dist_map_sz[0] // 2, dist_map_sz[1] // 2], dtype=tf.float32)
        center = tf.cast(center, reference_feat.dtype)

        #TODO: modify DistanceMap to generate the required tensor shape
        dist_map = self.distance_map(center, dist_map_sz)
        dist_map = tf.reshape(dist_map, [1, dist_map_sz[0], dist_map_sz[1], self.num_bins])

        # Compute target map, weights v_plus and weight_m (used in v_minus), used for reference loss
        #target_map = self.label_map_predictor(dist_map).reshape(1, -1, 1, 1)
        target_map = tf.reshape(self.label_map_predictor(dist_map), [1, -1, 1, 1])
        #print('target_map size is '+ str(target_map.shape))
        #v_plus = self.spatial_weight_predictor(dist_map).reshape(1, -1, 1, 1)
        v_plus = tf.reshape(self.spatial_weight_predictor(dist_map), [1, -1, 1, 1])
        #print('v_plus size is '+ str(v_plus.shape))
        #weight_m = self.target_mask_predictor(dist_map).reshape(1, -1, 1, 1)
        #weight_m = tf.reshape(self.target_mask_predictor(dist_map), [1, -1, 1, 1])

        #print('dist_map original size:' + str(dist_map.shape))
        weight_m = self.target_mask_predictor(dist_map)
        #self.target_mask_predictor.summary()
        #print("SUMMMMMMMMMMMMMMMMMMMMMMMMARY")

        #print('weight_m original size:' + str(weight_m.shape))
        weight_m = tf.reshape(weight_m, [1, -1, 1, 1])
        #print('weight_m final size:' + str(weight_m.shape))
        # compute regularizer term
        step_length = tf.math.exp(self.log_step_length)
        reg_weight = tf.clip_by_value((self.filter_reg*self.filter_reg), clip_value_min=self.min_filter_reg**2, clip_value_max=tf.float32.max) / (feat_dim**2)

        losses = {'train': [], 'train_reference_loss': [], 'train_reg': [], 'train_query_loss': []}

        for i in range(num_iter):
            # I. Computing gradient of reference loss with respect to the filter map
            # Computing the cost volume between the filter map and the reference features
            correlation_fn = correlation.FunctionCorrelation()
            filter_map = tf.reshape(filter_map, [filter_map.shape[0], filter_map.shape[3], filter_map.shape[1], filter_map.shape[2]])
            reference_feat = tf.reshape(reference_feat, [reference_feat.shape[0], reference_feat.shape[3], reference_feat.shape[1], reference_feat.shape[2]])
            scores_filter_w_ref = correlation_fn(filter_map, reference_feat)
            print('scores_filter_w_ref size: ' + str(scores_filter_w_ref.shape))
            print('filter_map size: ' + str(filter_map.shape))
            print('reference_feat size: ' + str(reference_feat.shape))

            # Computing Reference Frame Objective L_R and corresponding gradient with respect to the filter map
            # Applying sigma function on the score:
            act_scores_filter_w_ref = v_plus * self.score_activation(scores_filter_w_ref, weight_m)
            print('act_scores_filter_w_ref ' + str(act_scores_filter_w_ref.shape))
            print('scores_filter ' + str(scores_filter_w_ref.shape))
            print('weight_m ' + str(weight_m.shape))
            grad_act_scores_by_filter = v_plus * self.score_activation_deriv(scores_filter_w_ref, weight_m)
            print('grad_act_scores_by_filter ' + str(grad_act_scores_by_filter.shape))
            print('scores_filter ' + str(scores_filter_w_ref.shape))
            print('weight_m ' + str(weight_m.shape))
            loss_ref_residuals = act_scores_filter_w_ref - v_plus * target_map
            mapped_residuals = grad_act_scores_by_filter * loss_ref_residuals

            # Computing the gradient of the reference loss with respect to the filer map
            correlation_transpose_fn = correlation.FunctionCorrelationTranspose()
            filter_grad_loss_ref = correlation_transpose_fn(mapped_residuals, reference_feat)

            # Computing the gradient of the regularization term with respect to the filter map
            filter_grad_reg = reg_weight * filter_map

            filter_grad = filter_grad_reg + filter_grad_loss_ref

            if compute_losses:
                # compute corresponding loss
                loss_ref = 0.5 * tf.math.reduce_sum(loss_ref_residuals**2)/num_sequences
                loss_reg = 0.5 / reg_weight.item() * tf.math.reduce_sum(filter_grad_reg ** 2) / num_sequences

            # II. Computing Query Frame Objective L_q and corresponding gradient with respect to the filter map
            loss_query = 0
            #if self.apply_query_loss:
            if False:
                # Computing the cost volume between the filter map and the query features
                # dimension (b, search_size*search_size, H, W)
                scores_filter_w_query = FunctionCorrelation(filter_map, query_feat)

                # Applying the 4D kernel on the cost volume,
                loss_query_residuals = self.reg_layer(scores_filter_w_query.reshape(-1, self.search_size,
                                                                                    self.search_size, *feat_sz))
                # output shape is (b, H, W, output_dim, search_size, search_size)

                #  Computing the gradient of the query loss with respect to the filer map
                # apply transpose convolution, returns to b, search_size, search_size, H, W
                reg_tp_res = tf.reshape(self.reg_layer(loss_query_residuals, transpose=True), scores_filter_w_query.shape)

                filter_grad_loss_query = FunctionCorrelationTranspose(reg_tp_res, query_feat)
                filter_grad += filter_grad_loss_query
                if compute_losses:
                    # calculate the corresponding loss:
                    loss_query = 0.5 * tf.reduce_sum(loss_query_residuals ** 2) / num_sequences

            # III. Calculating alpha denominator
            # 1. Reference loss (L_r)
            # Computing the cost volume between the gradient of the loss with respect to the filter map with
            # the reference features in scores_filter_grad_w_ref
            scores_filter_grad_w_ref_fn = correlation.FunctionCorrelation()
            scores_filter_grad_w_ref = scores_filter_grad_w_ref_fn(filter_grad, reference_feat)
            scores_filter_grad_w_ref = grad_act_scores_by_filter * scores_filter_grad_w_ref
            if self.apply_query_loss:
                alpha_den = tf.reduce_sum(tf.reshape(scores_filter_grad_w_ref * scores_filter_grad_w_ref, [num_sequences, -1]), axis=1)
                # shape is b
            else:
                alpha_den = tf.reduce_sum(scores_filter_grad_w_ref * scores_filter_grad_w_ref, axis=1, keepdims=True)
                # shape is b, spa**2, H, W

            # 2. Query Loss (L_q)
            if self.apply_query_loss:
                # Hessian parts for regularization
                scores_filter_grad_w_query = FunctionCorrelation(filter_grad, query_feat)
                alpha_den_loss_query_residual = tf.reshape(self.reg_layer(scores_filter_grad_w_query), [-1,
                                                                                                  self.search_size,
                                                                                                  self.search_size,
                                                                                                  *feat_sz])
                alpha_den += tf.reduce_sum(tf.reshape((alpha_den_loss_query_residual * alpha_den_loss_query_residual),\
                                                      [num_sequences, -1]), axis=1)

            # IV. Compute step length alpha
            if self.apply_query_loss:
                alpha_num = tf.reduce_sum(tf.reshape((filter_grad * filter_grad), [num_sequences, -1]), axis=1)
            else:
                alpha_num = tf.reduce_sum((filter_grad * filter_grad), axis=1, keepdims=True)
            #alpha_den = (alpha_den + reg_weight * alpha_num).clamp(1e-8)
            alpha_den = tf.clip_by_value((alpha_den + reg_weight * alpha_num), clip_value_min=1e-8, clip_value_max=tf.float32.max)
            alpha = alpha_num / alpha_den

            # V. Update filter map
            if self.apply_query_loss:
                #filter_map = filter_map - (step_length * alpha.view(num_sequences, 1, 1, 1)) * filter_grad
                filter_map = filter_map - (step_length * tf.reshape(alpha, [num_sequences, 1, 1, 1])) * filter_grad
            else:
                filter_map = filter_map - (step_length * alpha) * filter_grad

            if compute_losses:
                losses['train_reference_loss'].append(loss_ref)
                losses['train_reg'].append(loss_reg)
                losses['train_query_loss'].append(loss_query)
                losses['train'].append(losses['train_reference_loss'][-1] + losses['train_reg'][-1] +
                                       losses['train_query_loss'][-1])

        if compute_losses:
            print('LocalGOCor: train reference loss is {}'.format(losses['train_reference_loss']))
            print('LocalGOCor: train query loss is {}'.format(losses['train_query_loss']))
            print('LocalGOCor: train reg is {}\n'.format(losses['train_reg']))

        return filter_map, losses


class LocalGOCor(tf.keras.layers.Layer):
    """The main LocalGOCor module for computing the local correlation volume.
    For now, only supports local search radius of 4. 
    args:
        filter_initializer: initializer network
        filter_optimizer: optimizer network
    """
    def __init__(self, filter_initializer, filter_optimizer):
        super(LocalGOCor, self).__init__()

        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer

    def call(self, reference_feat, query_feat, **kwargs):
        """
        Computes the local GOCor correspondence volume between inputted reference and query feature maps.
        args:
            reference_feat: reference feature with shape (b, feat_dim, H, W)
            query_feat: query feature with shape (b, feat_dim, H2, W2)

        output:
            scores: local correspondence volume between the optimized filter map (instead of the reference features in the
                    feature correlation layer) and the query feature map.
        """
        
        # initializes the filter map
        print('reference_feat shape ' + str(reference_feat.shape))
        filter = self.filter_initializer(reference_feat)
        print('filter shape ' + str(filter.shape))
        #filter = LocalCorrSimpleInitializer(reference_feat)
        
        # optimizes the filter map
        filter, losses = self.filter_optimizer(filter, reference_feat, query_feat=query_feat, **kwargs)
        
        # compute the local cost volume between optimized filter map and query features
        scores_fn = correlation.FunctionCorrelation()
        filter = tf.reshape(filter, [filter.shape[0], filter.shape[3], filter.shape[1], filter.shape[2]])
        query_feat = tf.reshape(query_feat, [query_feat.shape[0], query_feat.shape[3], query_feat.shape[1], query_feat.shape[2]])
        scores = scores_fn(filter, query_feat)
        print('JEN INPUT SIZE: ' + str(query_feat.shape))
        print('JEN FILTER SIZE: ' + str(filter.shape))
        print('JEN OUTPUT SIZE: ' + str(scores.shape))
        
        return scores

