# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
from torch.nn import BatchNorm2d
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_backbone, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.uda.masking_consistency_module import MaskingConsistencyModule
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio
from mmseg.models.uda.min_norm_solver import MinNormSolver


def _params_equal(model, student_1, student_2):
    for ema_param, param1, param2 in zip(model.named_parameters(), 
                                         student_1.named_parameters(), 
                                         student_2.named_parameters()):
        if not torch.equal(ema_param[1].data, param1[1].data) or not torch.equal(ema_param[1].data, param2[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DTDA(UDADecorator):

    def __init__(self, **cfg):
        super(DTDA, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.divergence_regulator = cfg['divergence_regulator']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.share_encoder = cfg['share_encoder']
        self.pretrained = cfg['pretrained']
        self.cfg = cfg
        self.ema_model_decoder_dup_norm = cfg['model'].decode_head.decoder_params.fusion_cfg.dup_norm
        self.ema_method = cfg['ema_method']
        assert self.ema_method in ['naive', 'min_norm']
        self.model1_ema_weight = 0.5
        self.model2_ema_weight = 0.5

        self.merge_buffer = cfg["merge_buffer"]
        assert self.merge_buffer in ["mean", "ema", False]
        assert not (self.ema_model_decoder_dup_norm and self.merge_buffer)
        assert self.mix == 'class'

        self.debug_fdist_mask_t1 = None
        self.debug_fdist_mask_t2 = None
        self.debug_gt_rescale_t1 = None
        self.debug_gt_rescale_t2 = None

        self.class_probs = {}

        student_1_cfg = deepcopy(cfg['model'])
        student_2_cfg = deepcopy(cfg['model'])
        if self.share_encoder:
            common_backbone = self._build_common_backbone(cfg)
            student_1_cfg['backbone'] = common_backbone
            student_2_cfg['backbone'] = common_backbone
        # self.model is now the ema model a.k.a. teacher model

        self.student_model_1 = build_segmentor(student_1_cfg)
        self.student_model_2 = build_segmentor(student_2_cfg)

        if self.enable_fdist:   
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        self.mic1, self.mic2 = None, None
        if self.enable_masking:
            self.mic1 = MaskingConsistencyModule(require_teacher=False, cfg=deepcopy(cfg))
            self.mic2 = MaskingConsistencyModule(require_teacher=False, cfg=deepcopy(cfg))

    # def get_ema_model(self):
    #     return get_module(self.ema_model)

    def _build_common_backbone(self, cfg):
        backbone = deepcopy(cfg['model']['backbone'])
        pretrained = cfg['model'].get('pretrained')
        if pretrained is not None:
            assert backbone.get('pretrained') is None, 'both backbone and segmentor set pretrained weight'
            backbone['pretrained'] = pretrained
        return build_backbone(backbone)

    def get_student_model(self, model: int):
        if model == 1:
            return get_module(self.student_model_1)
        elif model == 2:
            return get_module(self.student_model_2)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_student_weights(self, force_student_equal=True):  # doesn't matter which is set to equal to which? does teacher have to equal to student or is the effect same the other way around?
        for param in self.get_model().parameters():
            param.detach_()
        if force_student_equal:
            student1_p = list(self.get_student_model(1).parameters())
            student2_p = list(self.get_student_model(2).parameters())
            teacher_p = list(self.get_model().parameters())
            for i in range(0, len(teacher_p)):
                if not teacher_p[i].data.shape:  # scalar tensor
                    student1_p[i].data = teacher_p[i].data.clone()
                    student2_p[i].data = teacher_p[i].data.clone()
                else:
                    student1_p[i].data[:] = teacher_p[i].data[:].clone()
                    student2_p[i].data[:] = teacher_p[i].data[:].clone()

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight
    
    def _merge_bn_buffers(self, iter):
        assert self.ema_model_decoder_dup_norm == False
        student1_head_mean = ((name, buffer) for name, buffer in self.get_student_model(1).decode_head.head.fuse_layer.named_buffers() if "bn." in name and "running_mean" in name)
        student2_head_mean = ((name, buffer) for name, buffer in self.get_student_model(2).decode_head.head.fuse_layer.named_buffers() if "bn." in name and "running_mean" in name)
        student1_head_var = ((name, buffer) for name, buffer in self.get_student_model(1).decode_head.head.fuse_layer.named_buffers() if "bn." in name and "running_var" in name)
        student2_head_var = ((name, buffer) for name, buffer in self.get_student_model(2).decode_head.head.fuse_layer.named_buffers() if "bn." in name and "running_var" in name)
        student1_batches_tracked = ((name, buffer) for name, buffer in self.get_student_model(1).decode_head.head.fuse_layer.named_buffers() if "bn." in name and "batches_tracked" in name)
        student2_batches_tracked = ((name, buffer) for name, buffer in self.get_student_model(2).decode_head.head.fuse_layer.named_buffers() if "bn." in name and "batches_tracked" in name)
        student1_attention_mean = ((name, buffer) for name, buffer in self.get_student_model(1).decode_head.scale_attention.fuse_layer.named_buffers() if "bn." in name and "running_mean" in name)
        student2_attention_mean = ((name, buffer) for name, buffer in self.get_student_model(2).decode_head.scale_attention.fuse_layer.named_buffers() if "bn." in name and "running_mean" in name)
        student1_attention_var = ((name, buffer) for name, buffer in self.get_student_model(1).decode_head.scale_attention.fuse_layer.named_buffers() if "bn." in name and "running_var" in name)
        student2_attention_var = ((name, buffer) for name, buffer in self.get_student_model(2).decode_head.scale_attention.fuse_layer.named_buffers() if "bn." in name and "running_var" in name)
        student1_attention_batches_tracked = ((name, buffer) for name, buffer in self.get_student_model(1).decode_head.scale_attention.fuse_layer.named_buffers() if "bn." in name and "batches_tracked" in name)
        student2_attention_batches_tracked = ((name, buffer) for name, buffer in self.get_student_model(2).decode_head.scale_attention.fuse_layer.named_buffers() if "bn." in name and "batches_tracked" in name)

        debug1, debug2 = False, False

        if self.merge_buffer == "mean":
            for i, (ema_buffer_name, ema_buffer) in enumerate(self.get_model().decode_head.head.fuse_layer.named_buffers()):
                if "bn." in ema_buffer_name:
                    if "running_mean" in ema_buffer_name:
                        name1, buffer1 = next(student1_head_mean)
                        name2, buffer2 = next(student2_head_mean)
                        assert ema_buffer_name == name1 == name2, f"Merging buffers from different types of layers:\n{ema_buffer_name}\n{name1}\n{name2}"
                        ema_buffer.data = buffer1.data * 0.5 + buffer2.data * 0.5
                        debug1 = True
                    elif "running_var" in ema_buffer_name:
                        name1, buffer1 = next(student1_head_var)
                        name2, buffer2 = next(student2_head_var)
                        assert ema_buffer_name == name1 == name2, f"Merging buffers from different types of layers:\n{ema_buffer_name}\n{name1}\n{name2}"
                        ema_buffer.data = buffer1.data * 0.25 + buffer2.data * 0.25
                        debug1 = True
                    elif "batches_tracked" in ema_buffer_name:
                        name1, buffer1 = next(student1_batches_tracked)
                        name2, buffer2 = next(student2_batches_tracked)
                        assert ema_buffer_name == name1 == name2, f"Merging buffers from different types of layers:\n{ema_buffer_name}\n{name1}\n{name2}"
                        ema_buffer.data = buffer1.data * 0.5 + buffer2.data * 0.5
                        debug1 = True
                    else:
                        raise ModuleNotFoundError("Buffer must be one of running_mean, running_var, num_batches_tracked")
            
            for i, (ema_buffer_name, ema_buffer) in enumerate(self.get_model().decode_head.scale_attention.fuse_layer.named_buffers()):
                if "bn." in ema_buffer_name:
                    if "running_mean" in ema_buffer_name:
                        name1, buffer1 = next(student1_attention_mean)
                        name2, buffer2 = next(student2_attention_mean)
                        assert ema_buffer_name == name1 == name2, f"Merging buffers from different types of layers:\n{ema_buffer_name}\n{name1}\n{name2}"
                        ema_buffer.data = buffer1.data * 0.5 + buffer2.data * 0.5
                        debug2 = True
                    elif "running_var" in ema_buffer_name:
                        name1, buffer1 = next(student1_attention_var)
                        name2, buffer2 = next(student2_attention_var)
                        assert ema_buffer_name == name1 == name2, f"Merging buffers from different types of layers:\n{ema_buffer_name}\n{name1}\n{name2}"
                        ema_buffer.data = buffer1.data * 0.25 + buffer2.data * 0.25
                        debug2 = True
                    elif "batches_tracked" in ema_buffer_name:
                        name1, buffer1 = next(student1_attention_batches_tracked)
                        name2, buffer2 = next(student2_attention_batches_tracked)
                        assert ema_buffer_name == name1 == name2, f"Merging buffers from different types of layers:\n{ema_buffer_name}\n{name1}\n{name2}"
                        ema_buffer.data = buffer1.data * 0.5 + buffer2.data * 0.5
                        debug2 = True
            assert debug1 and debug2

        elif self.merge_buffer == "ema":
            alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
            for i, (ema_buffer_name, ema_buffer) in enumerate(self.get_model().decode_head.head.fuse_layer.named_buffers()):
                if "bn." in ema_buffer_name:
                    if "running_mean" in ema_buffer_name:
                        ema_buffer.data = alpha_teacher * ema_buffer.data + (1 - alpha_teacher) * self.model1_ema_weight * next(student1_head_mean)[1].data + (1 - alpha_teacher) * self.model2_ema_weight * next(student2_head_mean)[1].data
                    elif "running_var" in ema_buffer_name:
                        ema_buffer.data = alpha_teacher * ema_buffer.data + ((1 - alpha_teacher) / 2) * self.model1_ema_weight * next(student1_head_var)[1].data + ((1 - alpha_teacher) / 2) * self.model2_ema_weight * next(student2_head_var)[1].data
                    elif "batches_tracked" in ema_buffer_name:
                        ema_buffer.data = next(student1_batches_tracked)[1].data * 0.5 + next(student2_batches_tracked)[1].data * 0.5
            
            for i, (ema_buffer_name, ema_buffer) in enumerate(self.get_model().decode_head.scale_attention.fuse_layer.named_buffers()):
                if "bn." in ema_buffer_name:
                    if "running_mean" in ema_buffer_name:
                        ema_buffer.data = alpha_teacher * ema_buffer.data + (1 - alpha_teacher) * self.model1_ema_weight * next(student1_attention_mean)[1].data + (1 - alpha_teacher) * self.model2_ema_weight * next(student2_attention_mean)[1].data
                    elif "running_var" in ema_buffer_name:
                        ema_buffer.data = alpha_teacher * ema_buffer.data + ((1 - alpha_teacher) / 2) * self.model1_ema_weight * next(student1_attention_var)[1].data + ((1 - alpha_teacher) / 2) * self.model2_ema_weight * next(student2_attention_var)[1].data
                    elif "batches_tracked" in ema_buffer_name:
                        ema_buffer.data = next(student1_attention_batches_tracked)[1].data * 0.5 + next(student2_attention_batches_tracked)[1].data * 0.5

    def _copy_bn_buffers(self):
        assert self.ema_model_decoder_dup_norm == 2
        student1_head_buffers = ((name, buffer) for name, buffer in self.get_student_model(1).decode_head.head.fuse_layer.named_buffers() if "bn0." in name)
        student2_head_buffers = ((name, buffer) for name, buffer in self.get_student_model(2).decode_head.head.fuse_layer.named_buffers() if "bn0." in name)
        for i, (ema_buffer_name, ema_buffer) in enumerate(self.get_model().decode_head.head.fuse_layer.named_buffers()):
            if "bn0." in ema_buffer_name:
                name, buffer = next(student1_head_buffers)
            elif "bn1." in ema_buffer_name:
                name, buffer = next(student2_head_buffers)
                ema_buffer_name = ema_buffer_name.replace("bn1.", "bn0.")
            else:
                raise NotImplementedError(f"Buffer not recognized from {ema_buffer_name}")
            assert name == ema_buffer_name
            ema_buffer.data = buffer.data.clone()

        student1_scale_attention_buffers = ((name, buffer) for name, buffer in self.get_student_model(1).decode_head.scale_attention.fuse_layer.named_buffers() if "bn0." in name)
        student2_scale_attention_buffers = ((name, buffer) for name, buffer in self.get_student_model(2).decode_head.scale_attention.fuse_layer.named_buffers() if "bn0." in name)
        for i, (ema_buffer_name, ema_buffer) in enumerate(self.get_model().decode_head.scale_attention.fuse_layer.named_buffers()):
            if "bn0." in ema_buffer_name:
                name, buffer = next(student1_scale_attention_buffers)
            elif "bn1." in ema_buffer_name:
                name, buffer = next(student2_scale_attention_buffers)
                ema_buffer_name = ema_buffer_name.replace("bn1.", "bn0.")
            else:
                raise NotImplementedError(f"Buffer not recognized from {ema_buffer_name}")
            assert name == ema_buffer_name
            ema_buffer.data = buffer.data.clone()
        
        buffer_ids = []
        for buffer in self.get_model().decode_head.scale_attention.fuse_layer.buffers():
            buffer_ids.append(id(buffer))
        assert buffer_ids == self.buffer_ids

    def _find_min_norm_weight(self):
        grads = {1: [], 2: []}

        for ema_param, param_1, param_2 in zip(
                            self.get_model().parameters(), 
                            self.get_student_model(1).parameters(),
                            self.get_student_model(2).parameters()):
            if not ema_param.data.shape:
                grads[1].append((param_1.data - ema_param.data).clone().detach())
                grads[2].append((param_2.data - ema_param.data).clone().detach())
            else:
                grads[1].append((param_1[:].data[:] - ema_param[:].data[:]).clone().detach())
                grads[2].append((param_2[:].data[:] - ema_param[:].data[:]).clone().detach())
        weights, _ = MinNormSolver.find_min_norm_element([grads[i] for i in [1, 2]])
        self.model1_ema_weight = weights[0]
        self.model2_ema_weight = weights[1]
        print(f"model1 ema weight: {self.model1_ema_weight}\tmodel2 ema weight: {self.model2_ema_weight}")

    def _update_ema(self, iter, weights='naive'):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)

        for ema_param, param_1, param_2 in zip(
            self.get_model().parameters(), 
            self.get_student_model(1).parameters(),
            self.get_student_model(2).parameters()
            ):
            if not ema_param.data.shape:  # scalar tensor
                ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * self.model1_ema_weight * param_1.data + (1 - alpha_teacher) * self.model2_ema_weight * param_2.data
            else:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * self.model1_ema_weight * param_1[:].data[:] + (1 - alpha_teacher) * self.model2_ema_weight * param_2[:].data[:]

    def _norm_regulate(self, method):  # L2 norm loss
        cumulative_diff = 0
        for param1, param2 in zip(
            self.get_student_model(1).parameters(),
            self.get_student_model(2).parameters()
            ):
            # If dim= None and ord= None, A will be flattened to 1D and the 2-norm of the resulting vector will be computed.
            if method == 'l1':
                cumulative_diff += torch.sum((param1 - param2).abs())
            elif method == 'l2':
                cumulative_diff += torch.sum((param1 - param2) ** 2)
            else:
                raise NotImplementedError()
            # l2_diff += torch.linalg.norm((param1 - param2))
        cumulative_diff /= sum([torch.numel(i) for i in self.get_student_model(1).parameters()])  # divide by total number of elements
        if method == 'l2':
            cumulative_diff = cumulative_diff ** (1 / 2)
        cumulative_diff *= self.divergence_regulator['weight']
        return cumulative_diff  # miniminze Euclidean distance
    
    def _cos_regulate(self, method):  # cosine similarity loss (maximize)
        similarity_sum = 0
        for param1, param2 in zip(
            self.get_student_model(1).parameters(),
            self.get_student_model(2).parameters()
            ):
            # Flatten the parameters and add a batch dimension
            param1_flat = param1.view(1, -1)
            param2_flat = param2.view(1, -1)
            if method == 'min':
                similarity_sum += (1 - F.cosine_similarity(param1_flat, param2_flat))
            elif method == 'orth':
                similarity_sum += (F.cosine_similarity(param1_flat, param2_flat)**2)
            else:
                raise NotImplementedError()
        similarity_sum *= self.divergence_regulator['weight']
        return similarity_sum
    
    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, student_id, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_student_model(student_id), HRDAEncoderDecoder) and self.get_student_model(student_id).feature_scale in self.get_student_model(student_id).feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        if student_id == 1:
                            self.debug_fdist_mask_t1 = fdist_mask
                            self.debug_gt_rescale_t1 = gt_rescaled
                        elif student_id == 2:
                            self.debug_fdist_mask_t2 = fdist_mask
                            self.debug_gt_rescale_t2 = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                if student_id == 1:
                    self.debug_fdist_mask_t1 = fdist_mask
                    self.debug_gt_rescale_t1 = gt_rescaled
                elif student_id == 2:
                    self.debug_fdist_mask_t2 = fdist_mask
                    self.debug_gt_rescale_t2 = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        if self.local_iter % self.debug_img_interval == 0:
            self.get_model().decode_head.debug = True
            self.get_student_model(1).decode_head.debug = True
            self.get_student_model(2).decode_head.debug = True
        else:
            self.get_model().decode_head.debug = False
            self.get_student_model(2).decode_head.debug = False
            self.get_student_model(2).decode_head.debug = False

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target1_img,
                      target1_img_metas,
                      target2_img,
                      target2_img_metas,
                      rare_class=None,
                      target1_valid_pseudo_mask=None,
                      target2_valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init student model/update teacher model with ema
        if self.local_iter == 0:
            if self.pretrained:
                self.load_state_dict(torch.load(self.pretrained)['state_dict'])
                self._init_student_weights(force_student_equal=False)
                print(f"######\nLoaded state dict from {self.pretrained}\n######")
            else:
                self._init_student_weights()
                assert _params_equal(self.get_model(), self.get_student_model(1), self.get_student_model(2))
                print("Params equal check passed")

            self.buffer_ids = []
            for buffer in self.get_model().decode_head.scale_attention.fuse_layer.buffers():
                self.buffer_ids.append(id(buffer))

        if self.local_iter > 0:
            if self.ema_method == 'min_norm':
                self._find_min_norm_weight()
            self._update_ema(self.local_iter)
            # if self.ema_model_decoder_dup_norm:
            #     self._copy_bn_buffers()
            # elif self.merge_buffer:
            #     self._merge_bn_buffers(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        if self.mic1 is not None:
            self.mic1.update_weights(self.get_student_model(1), self.local_iter)
        if self.mic2 is not None:
            self.mic2.update_weights(self.get_student_model(2), self.local_iter)

        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images, student 1
        clean_losses = self.get_student_model(1).forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_losses = add_prefix(clean_losses, 'student1_source')
        seg_debug['Student 1 Source'] = self.get_student_model(1).decode_head.debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_student_model(1).backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance, student 1
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, student_id=1, feat=src_feat)
            log_vars.update(add_prefix(feat_log, 'student1_fdist'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

            del feat_loss
        del src_feat, clean_loss, clean_losses
        
        # Train on source images, student 2
        clean_losses = self.get_student_model(2).forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_losses = add_prefix(clean_losses, 'student2_source')
        seg_debug['Student 2 Source'] = self.get_student_model(2).decode_head.debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_student_model(2).backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance, student 2
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, student_id=2, feat=src_feat)
            log_vars.update(add_prefix(feat_log, 'student2_fdist'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

            del feat_loss
        del src_feat, clean_loss, clean_losses

        # Generate pseudo-label
        for m in self.get_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        
        # if self.ema_model_decoder_dup_norm or self.merge_buffer:
        #     for m in self.get_model().decode_head.head.fuse_layer.modules():
        #         if isinstance(m, BatchNorm2d):
        #             m.training = False

        #     for m in self.get_model().decode_head.scale_attention.fuse_layer.modules():
        #         if isinstance(m, BatchNorm2d):
        #             m.training = False

        # pseudo-label for student 1
        pseudo_label_t1, pseudo_weight_t1 = None, None
        ema_logits_t1 = self.get_model().generate_pseudo_label(
            target1_img, target1_img_metas, decoder_norm_select=0)
        seg_debug['Target 1'] = self.get_model().decode_head.debug_output
        pseudo_label_t1, pseudo_weight_t1 = self.get_pseudo_label_and_weight(ema_logits_t1)
        pseudo_weight_t1 = self.filter_valid_pseudo_region(pseudo_weight_t1, target1_valid_pseudo_mask)
        gt_pixel_weight_t1 = torch.ones((pseudo_weight_t1.shape), device=dev)
        del ema_logits_t1
        # ema_softmax_t1 = torch.softmax(ema_logits_t1.detach(), dim=1)
        # del ema_logits_t1

        # pseudo_prob_t1, pseudo_label_t1 = torch.max(ema_softmax_t1, dim=1)
        # ps_large_p_t1 = pseudo_prob_t1.ge(self.pseudo_threshold).long() == 1
        # ps_size_t1 = np.size(np.array(pseudo_label_t1.cpu()))
        # pseudo_weight_t1 = torch.sum(ps_large_p_t1).item() / ps_size_t1
        # pseudo_weight_t1 = pseudo_weight_t1 * torch.ones(
        #     pseudo_prob_t1.shape, device=dev)
        # del pseudo_prob_t1, ps_large_p_t1, ps_size_t1

        # pseudo-label for student 2
        ema_logits_t2 = self.get_model().generate_pseudo_label(
            target2_img, target2_img_metas, decoder_norm_select=1 if self.ema_model_decoder_dup_norm else 0)
        seg_debug['Target 2'] = self.get_model().decode_head.debug_output
        pseudo_label_t2, pseudo_weight_t2 = self.get_pseudo_label_and_weight(ema_logits_t2)
        pseudo_weight_t2 = self.filter_valid_pseudo_region(pseudo_weight_t2, target2_valid_pseudo_mask)
        gt_pixel_weight_t2 = torch.ones((pseudo_weight_t2.shape), device=dev)
        del ema_logits_t2

        # pseudo_prob_t2, pseudo_label_t2 = torch.max(ema_softmax_t2, dim=1)
        # ps_large_p_t2 = pseudo_prob_t2.ge(self.pseudo_threshold).long() == 1
        # ps_size_t2 = np.size(np.array(pseudo_label_t2.cpu()))
        # pseudo_weight_t2 = torch.sum(ps_large_p_t2).item() / ps_size_t2
        # pseudo_weight_t2 = pseudo_weight_t2 * torch.ones(
        #     pseudo_prob_t2.shape, device=dev)
        # del pseudo_prob_t2, ps_large_p_t2, ps_size_t2

        # if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
        #     assert target1_valid_pseudo_mask is None and target2_valid_pseudo_mask is None
        #     pseudo_weight_t1[:, :self.psweight_ignore_top, :] = 0
        #     pseudo_weight_t2[:, :self.psweight_ignore_top, :] = 0
        # if self.psweight_ignore_bottom > 0:
        #     assert target1_valid_pseudo_mask is None and target2_valid_pseudo_mask is None
        #     pseudo_weight_t1[:, -self.psweight_ignore_bottom:, :] = 0
        #     pseudo_weight_t2[:, -self.psweight_ignore_bottom:, :] = 0
        # if target1_valid_pseudo_mask is not None:
        #     pseudo_weight_t1 *= target1_valid_pseudo_mask.squeeze(1)
        # if target2_valid_pseudo_mask is not None:
        #     pseudo_weight_t2 *= target2_valid_pseudo_mask.squeeze(1)
        # gt_pixel_weight_t1 = torch.ones((pseudo_weight_t1.shape), device=dev)
        # gt_pixel_weight_t2 = torch.ones((pseudo_weight_t2.shape), device=dev)

        # prepared variables at this point: pseudo_label_t1, pseudo_weight_t1, pseudo_label_t2, pseudo_weight_t2
        # gt_pixel_weight_t1, gt_pixel_weight_t2

        # Mixing Student 1
        mixed_img_t1, mixed_lbl_t1 = [None] * batch_size, [None] * batch_size
        mixed_seg_weight_t1 = pseudo_weight_t1.clone()
        mix_masks_t1 = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks_t1[i]
            mixed_img_t1[i], mixed_lbl_t1[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target1_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label_t1[i])))
            _, mixed_seg_weight_t1[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight_t1[i], pseudo_weight_t1[i])))
        del gt_pixel_weight_t1
        mixed_img_t1 = torch.cat(mixed_img_t1)
        mixed_lbl_t1 = torch.cat(mixed_lbl_t1)

        # Train on mixed images student 1
        mix_losses_t1 = self.get_student_model(1).forward_train(
            mixed_img_t1, img_metas, mixed_lbl_t1, mixed_seg_weight_t1, return_feat=False)
        seg_debug['Student 1 Mix'] = self.get_student_model(1).decode_head.debug_output
        mix_losses_t1 = add_prefix(mix_losses_t1, 'student1_mix')
        mix_loss_t1, mix_log_vars = self._parse_losses(mix_losses_t1)
        log_vars.update(mix_log_vars)
        mix_loss_t1.backward()

        # Mixing Student 2
        mixed_img_t2, mixed_lbl_t2 = [None] * batch_size, [None] * batch_size
        mixed_seg_weight_t2 = pseudo_weight_t2.clone()
        mix_masks_t2 = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks_t2[i]
            mixed_img_t2[i], mixed_lbl_t2[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target2_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label_t2[i])))
            _, mixed_seg_weight_t2[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight_t2[i], pseudo_weight_t2[i])))
        del gt_pixel_weight_t2
        mixed_img_t2 = torch.cat(mixed_img_t2)
        mixed_lbl_t2 = torch.cat(mixed_lbl_t2)

        # Train on mixed images student 2
        mix_losses_t2 = self.get_student_model(2).forward_train(
            mixed_img_t2, img_metas, mixed_lbl_t2, mixed_seg_weight_t2, return_feat=False)
        seg_debug['Student 2 Mix'] = self.get_student_model(2).decode_head.debug_output
        mix_losses_t2 = add_prefix(mix_losses_t2, 'student2_mix')
        mix_loss_t2, mix_log_vars = self._parse_losses(mix_losses_t2)
        log_vars.update(mix_log_vars)
        mix_loss_t2.backward()

        # regulate parameter space of student 1 model and student 2 model
        if self.divergence_regulator:
            if self.divergence_regulator['type'] == 'norm':
                method = self.divergence_regulator['method']
                norm_losses = self._norm_regulate(method)
                norm_loss, norm_log_vars = self._parse_losses({f'norm_{method}_loss': norm_losses})
                log_vars.update(norm_log_vars)
                norm_loss.backward()
                del norm_loss, norm_losses
            elif self.divergence_regulator['type'] == 'cos':
                method = self.divergence_regulator['method']
                cos_losses = self._cos_regulate(method)
                cos_loss, cos_log_vars = self._parse_losses({f'cos_{method}_loss': cos_losses})
                log_vars.update(cos_log_vars)
                cos_loss.backward()
                del cos_loss, cos_losses
            else:
                raise NotImplementedError()
        
        # Maked Training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss_t1 = self.mic1(self.get_student_model(1), img, img_metas,
                                   gt_semantic_seg, target1_img,
                                   target1_img_metas, target1_valid_pseudo_mask,
                                   pseudo_label_t1, pseudo_weight_t1)
            masked_loss_t2 = self.mic2(self.get_student_model(2), img, img_metas,
                                   gt_semantic_seg, target2_img,
                                   target2_img_metas, target2_valid_pseudo_mask,
                                   pseudo_label_t2, pseudo_weight_t2)
            seg_debug.update(self.mic1.debug_output)
            seg_debug.update(self.mic2.debug_output)
            masked_loss_t1 = add_prefix(masked_loss_t1, 'masked')
            masked_loss_t2 = add_prefix(masked_loss_t2, 'masked')
            masked_loss_t1, masked_log_vars_t1 = self._parse_losses(masked_loss_t1)
            masked_loss_t2, masked_log_vars_t2 = self._parse_losses(masked_loss_t2)
            log_vars.update(masked_log_vars_t1)
            log_vars.update(masked_log_vars_t2)
            masked_loss_t1.backward()
            masked_loss_t2.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_t1_img = torch.clamp(denorm(target1_img, means, stds), 0, 1)
            vis_t2_img = torch.clamp(denorm(target2_img, means, stds), 0, 1)
            vis_mixed_t1_img = torch.clamp(denorm(mixed_img_t1, means, stds), 0, 1)
            vis_mixed_t2_img = torch.clamp(denorm(mixed_img_t2, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 3, 6
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[0][1], gt_semantic_seg[j],'Source Seg GT', cmap='cityscapes')
                subplotimg(axs[0][2], pseudo_weight_t1[j], 'Pseudo W. Target 1', vmin=0, vmax=1)
                subplotimg(axs[0][3], pseudo_weight_t2[j], 'Pseudo W. Target 2', vmin=0, vmax=1)
                subplotimg(axs[1][0], vis_t1_img[j], 'Image Target 1')
                subplotimg(axs[1][1], pseudo_label_t1[j], 'Seg (Pseudo) GT Target 1',cmap='cityscapes')
                subplotimg(axs[1][2], vis_mixed_t1_img[j], 'Mixed Image Target 1')
                subplotimg(axs[1][3], mixed_lbl_t1[j], 'Seg (Mixed) GT Target 1', cmap='cityscapes')
                subplotimg(axs[1][4], mix_masks_t1[j][0], 'Domain Mask Target 1', cmap='gray')
                subplotimg(axs[2][0], vis_t2_img[j], 'Image Target 2')
                subplotimg(axs[2][1], pseudo_label_t2[j], 'Seg (Pseudo) GT Target 2',cmap='cityscapes')
                subplotimg(axs[2][2], vis_mixed_t2_img[j], 'Mixed Image Target 2')
                subplotimg(axs[2][3], mixed_lbl_t2[j], 'Seg (Mixed) GT Target 2', cmap='cityscapes')
                subplotimg(axs[2][4], mix_masks_t2[j][0], 'Domain Mask Target 2', cmap='gray')
                if self.debug_fdist_mask_t1 is not None:
                    subplotimg(axs[0][4], self.debug_fdist_mask_t1[j][0], 'FDist Mask Target 1', cmap='gray')
                if self.debug_fdist_mask_t2 is not None:
                    subplotimg(axs[0][5], self.debug_fdist_mask_t2[j][0], 'FDist Mask Target 2', cmap='gray')
                if self.debug_gt_rescale_t1 is not None:
                    subplotimg(axs[1][5], self.debug_gt_rescale_t1[j], 'Scaled GT Target 1', cmap='cityscapes')
                if self.debug_gt_rescale_t2 is not None:
                    subplotimg(axs[2][5], self.debug_gt_rescale_t2[j], 'Scaled GT Target 2', cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
            
            if seg_debug['Student 1 Source'] is not None and seg_debug:
                for j in range(batch_size):
                    rows, cols = 6, len(seg_debug['Student 1 Source'])
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            if out.shape[1] == 3:
                                vis = torch.clamp(denorm(out, means, stds), 0, 1)
                                subplotimg(axs[k1][k2], vis[j], f'{n1} {n2}')
                            else:
                                if out.ndim == 3:
                                    args = dict(cmap='cityscapes')
                                else:
                                    args = dict(cmap='gray', vmin=0, vmax=1)
                                subplotimg(axs[k1][k2], out[j], f'{n1} {n2}', **args)
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}_s.png'))
                    plt.close()
        self.local_iter += 1

        return log_vars
    
    def simple_test(self, img, img_meta, rescale=True, decoder_norm_select=0):
        """Simple test with single image."""
        # print(f"Using norm {decoder_norm_select} in decoder head")
        if self.eval_model == 'teacher':
            import pdb; breakpoint()
            return self.get_model().simple_test(img, img_meta, rescale, decoder_norm_select=decoder_norm_select)
        elif self.eval_model == 'student1':
            return self.get_student_model(1).simple_test(img, img_meta, rescale, decoder_norm_select=decoder_norm_select)
        elif self.eval_model == 'student2':
            return self.get_student_model(2).simple_test(img, img_meta, rescale, decoder_norm_select=decoder_norm_select)

    def aug_test(self, imgs, img_metas, rescale=True, decoder_norm_select=0):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # print(f"Using norm {decoder_norm_select} in decoder head")
        if self.eval_model == 'teacher':
            return self.get_model().aug_test(imgs, img_metas, rescale, decoder_norm_select=decoder_norm_select)
        elif self.eval_model == 'student1':
            return self.get_student_model(1).aug_test(imgs, img_metas, rescale, decoder_norm_select=decoder_norm_select)
        elif self.eval_model == 'student2':
            return self.get_student_model(2).aug_test(imgs, img_metas, rescale, decoder_norm_select=decoder_norm_select)
