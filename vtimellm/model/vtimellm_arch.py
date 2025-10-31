import re
import torch
import torch.nn as nn
from vtimellm.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from vtimellm.transformer_encoder_droppath import Transformer, build_position_encoding, init_weights
from abc import ABC, abstractmethod
import torch.nn.functional as F

import einops
import numpy as np

from .multimodal_encoder.clip_encoder import build_vision_tower

def temporal_iou(spans1, spans2):
    """
    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        iou: (N, M) torch.Tensor
        union: (N, M) torch.Tensor
    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> temporal_iou(test_spans1, test_spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = torch.max(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.min(spans1[:, None, 1], spans2[:, 1])  # (N, M

    inter = (right - left).clamp(min=0)  # (N, M)
    union = areas1[:, None] + areas2 - inter  # (N, M)

    iou = inter / union
    return iou, union

def generalized_temporal_iou(spans1, spans2):
    """
    Generalized IoU from https://giou.stanford.edu/
    Also reference to DETR implementation of generalized_box_iou
    https://github.com/facebookresearch/detr/blob/master/util/box_ops.py#L40

    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span in xx format [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        giou: (N, M) torch.Tensor

    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> generalized_temporal_iou(test_spans1, test_spans2)
    tensor([[ 0.6667,  0.2000],
        [-0.2000,  0.5000]])
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    assert (spans1[:, 1] >= spans1[:, 0]).all()
    assert (spans2[:, 1] >= spans2[:, 0]).all()
    iou, union = temporal_iou(spans1, spans2)

    left = torch.min(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.max(spans1[:, None, 1], spans2[:, 1])  # (N, M)
    enclosing_area = (right - left).clamp(min=0)  # (N, M)

    return iou - (enclosing_area - union) / enclosing_area


class Conv(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1, groups=1, bias=True, padding_mode='zeros')
                                    for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        x = x.permute(0,2,1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.permute(0, 2, 1)

class PatchFilter(nn.Module):

    def __init__(self, num_cls=100, num_patch=0, patch_filter_type='only_cls'):
        super().__init__()
        self.num_cls = num_cls
        self.patch_filter_type = patch_filter_type
        
    def forward(self, image):

        if len(image.shape) == 4: 
            if self.patch_filter_type == 'only_cls':
                image_patch = image[:, :, 1:, :]
                image = image[:, :, 0, :]
                num_PaddingToken =image.shape[1]
            elif self.patch_filter_type == 'cls_patch':
                pooling_method = 'compress_class_pool'  # 'compress_pool' 'mean_pool' 'compress_class_pool'

                if pooling_method == 'mean_pool':
                    pool_size = 2
                    bs = image.shape[0]
                    image_patch = image[:, :, 1:, :] 
                    image = image[:, :, 0, :]  

                    selected_frames = np.round(np.linspace(0, image_patch.shape[1] - 1, pool_size * pool_size)).astype(int)  
                    image_patch = image_patch[:, selected_frames, ...]  
                    image_patch = einops.rearrange(image_patch, 'b t (h w) d -> (b t) d h w', h=16, w=16)
                    image_patch = nn.functional.avg_pool2d(image_patch.float(), kernel_size=pool_size)
                    image_patch = einops.rearrange(image_patch.to(image.dtype), '(b t) d h w -> b (t h w) d', b=bs)

                    image = torch.cat([image, image_patch], dim=1)
                    num_PaddingToken =image.shape[1]

                elif pooling_method == 'compress_pool':
                    n_GoP = 4
                    s_GoP = 64
                    sim_gap = 0.7

                    bs = image.shape[0]
                    image_patch = image[:, :, 1:, :]
                    image = image[:, :, 0, :]
                    selected_frames = np.round(np.linspace(0, image_patch.shape[1] - 1, n_GoP)).astype(int)

                    compress_patches = []
                    for bs in range(image.shape[0]):
                        image_bs = image[bs]  
                        image_patch_bs = image_patch[bs]  
                        GoP_patches = []

                        for start_idx in selected_frames:
                            select_image_cls = image_bs[start_idx, ...]  
                            if start_idx + 3 > len(image_patch_bs):
                                start_idx = len(image_patch_bs)-3
                                end_idx = len(image_patch_bs)
                            else:
                                end_idx = start_idx + 3
                            select_image_patch = image_patch_bs[start_idx:end_idx, ...]
                            sim = F.cosine_similarity(select_image_patch[0].unsqueeze(0).repeat_interleave(len(select_image_patch[1:]), dim=0),
                                                      select_image_patch[1:],
                                                      dim=-1,)
                            
                            keep_mask = sim.flatten(0, 1) < sim_gap
                            kept_tokens = select_image_patch[1:].flatten(0, 1)[keep_mask]
                            IDR_tokens = select_image_patch[0]
                            cls_vs_idr_sim = F.cosine_similarity(IDR_tokens, 
                                                                 select_image_cls.unsqueeze(0).expand(IDR_tokens.shape[0], -1), 
                                                                 dim=-1)  

                            topk_sim, topk_idx = torch.topk(cls_vs_idr_sim, k=min(s_GoP, IDR_tokens.shape[0]), dim=-1)
                            cluster_centers = IDR_tokens[topk_idx]

                            if kept_tokens.numel() == 0:
                                kept_tokens_safe = kept_tokens.reshape(0, IDR_tokens.shape[-1])
                            else:
                                kept_tokens_safe = kept_tokens

                            tokens_to_assign_all = torch.cat([IDR_tokens, kept_tokens_safe], dim=0)  
                            centers_norm = F.normalize(cluster_centers, dim=-1)          
                            tokens_norm  = F.normalize(tokens_to_assign_all, dim=-1)     
                            sim_to_centers = tokens_norm @ centers_norm.t()              
                            assigned_center_idx = torch.argmax(sim_to_centers, dim=-1)   

                            pooled_features = []
                            for c in range(cluster_centers.shape[0]):
                                member_mask = (assigned_center_idx == c)
                                if member_mask.any():
                                    members = tokens_to_assign_all[member_mask]
                                    pooled = members.mean(dim=0)
                                else:
                                    pooled = cluster_centers[c]
                                pooled_features.append(pooled)
                            pooled_features = torch.stack(pooled_features, dim=0) 
                            GoP_patches.append(pooled_features)
                        compress_patches.append(torch.cat(GoP_patches, dim=0))

                    compress_patches = torch.stack(compress_patches, dim=0)
                    image = torch.cat([image, compress_patches], dim=1)
                    num_PaddingToken = image.shape[1]

                elif pooling_method == 'compress_class_pool':
                    n_GoP = 4
                    s_GoP = 64

                    bs = image.shape[0]
                    image_patch = image[:, :, 1:, :]
                    image = image[:, :, 0, :]
                    selected_frames = np.round(np.linspace(0, image_patch.shape[1] - 1, n_GoP)).astype(int)

                    compress_patches = []
                    for bs in range(image.shape[0]):
                        image_bs = image[bs]
                        image_patch_bs = image_patch[bs]
                        GoP_patches = []

                        for start_idx in selected_frames:
                            select_image_cls = image_bs[start_idx, ...]
                            if start_idx + 3 > len(image_patch_bs):
                                start_idx = len(image_patch_bs)-3
                                end_idx = len(image_patch_bs)
                            else:
                                end_idx = start_idx + 3
                            
                            select_image_patch = image_patch_bs[start_idx:end_idx, ...]
                            IDR_tokens = select_image_patch[0]
                            num_idr_tokens = IDR_tokens.shape[0]
                            cls_vs_idr_sim = F.cosine_similarity(
                                IDR_tokens,
                                select_image_cls.unsqueeze(0).expand(num_idr_tokens, -1),
                                dim=-1
                            )

                            learn_k = max(s_GoP - 36, 0)
                            learn_k = min(learn_k, num_idr_tokens)
                            if learn_k > 0:
                                _, topk_idx = torch.topk(cls_vs_idr_sim, k=learn_k, dim=-1)
                                topk_idx_sorted, _ = torch.sort(topk_idx)
                                selected_learn_idx = topk_idx_sorted
                            else:
                                selected_learn_idx = torch.tensor([], dtype=torch.long, device=IDR_tokens.device)

                            all_idx = torch.arange(num_idr_tokens, device=IDR_tokens.device)
                            if selected_learn_idx.numel() > 0:
                                remain_mask = torch.ones(num_idr_tokens, dtype=torch.bool, device=IDR_tokens.device)
                                remain_mask[selected_learn_idx] = False
                                remaining_idx = all_idx[remain_mask]
                            else:
                                remaining_idx = all_idx

                            sample_k = min(36, max(s_GoP - learn_k, 0), remaining_idx.numel())
                            if sample_k > 0 and remaining_idx.numel() > 0:
                                if sample_k == remaining_idx.numel():
                                    sampled_idx = remaining_idx
                                else:
                                    linspace_pos = torch.linspace(0, remaining_idx.numel() - 1, steps=sample_k, device=IDR_tokens.device)
                                    sampled_idx = remaining_idx[linspace_pos.round().long().unique(sorted=True)]
                                    if sampled_idx.numel() < sample_k:
                                        need = sample_k - sampled_idx.numel()
                                        extra = remaining_idx[:need]
                                        sampled_idx = torch.cat([sampled_idx, extra], dim=0)
                            else:
                                sampled_idx = torch.tensor([], dtype=torch.long, device=IDR_tokens.device)

                            center_idx = torch.cat([selected_learn_idx, sampled_idx], dim=0)
                            if center_idx.numel() > s_GoP:
                                center_idx = center_idx[:s_GoP]
                            if center_idx.numel() < s_GoP:
                                need = s_GoP - center_idx.numel()
                                full_sorted = torch.argsort(cls_vs_idr_sim, descending=True)
                                taken_mask = torch.ones_like(full_sorted, dtype=torch.bool)
                                if center_idx.numel() > 0:
                                    taken_mask[(full_sorted.unsqueeze(1) == center_idx.unsqueeze(0)).any(dim=1)] = False
                                candidates = full_sorted[taken_mask]
                                if candidates.numel() > 0:
                                    add_idx = candidates[:min(need, candidates.numel())]
                                    center_idx = torch.cat([center_idx, add_idx], dim=0)
                                if center_idx.numel() < s_GoP:
                                    pad_rep = center_idx[-1].repeat(s_GoP - center_idx.numel()) if center_idx.numel() > 0 else torch.zeros(s_GoP, dtype=torch.long, device=IDR_tokens.device)
                                    center_idx = torch.cat([center_idx, pad_rep], dim=0)
                            cluster_centers = IDR_tokens[center_idx]  
                            tokens_all = select_image_patch.reshape(-1, select_image_patch.shape[-1])  
                            centers_norm = F.normalize(cluster_centers, dim=-1)
                            tokens_norm = F.normalize(tokens_all, dim=-1)
                            sim_to_centers = tokens_norm @ centers_norm.t()  
                            labels_all = torch.argmax(sim_to_centers, dim=-1)  

                            labels_3x256 = labels_all.view(3, 256)
                            labels_f0 = labels_3x256[0]
                            keep_f1 = labels_3x256[1] != labels_f0
                            keep_f2 = labels_3x256[2] != labels_f0
                            kept_tokens_label = torch.cat([
                                select_image_patch[1][keep_f1],
                                select_image_patch[2][keep_f2]
                            ], dim=0)  
                            kept_labels = torch.cat([
                                labels_3x256[1][keep_f1],
                                labels_3x256[2][keep_f2]
                            ], dim=0)  
                            idr_tokens = select_image_patch[0]  
                            idr_labels = labels_3x256[0]        
                            pooled_features = []
                            for c in range(cluster_centers.shape[0]):
                                mask_idr = (idr_labels == c)
                                members = []
                                if mask_idr.any():
                                    members.append(idr_tokens[mask_idr])
                                if kept_tokens_label.shape[0] > 0:
                                    mask_kept = (kept_labels == c)
                                    if mask_kept.any():
                                        members.append(kept_tokens_label[mask_kept])
                                if len(members) == 0:
                                    pooled = cluster_centers[c]
                                else:
                                    pooled = torch.cat(members, dim=0).mean(dim=0)
                                pooled_features.append(pooled)
                            pooled_features = torch.stack(pooled_features, dim=0)
                            GoP_patches.append(pooled_features)
                        compress_patches.append(torch.cat(GoP_patches, dim=0))
                    compress_patches = torch.stack(compress_patches, dim=0)
                    image = torch.cat([image, compress_patches], dim=1)
                    num_PaddingToken = image.shape[1]

        elif len(image.shape) == 3:
            image = image
            num_PaddingToken = image.shape[1]

        return image, num_PaddingToken

class DetCriterion(nn.Module):

    def __init__(self, weight_dict, losses, span_loss_type='l1', eos_coef=0.1):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.temperature = 0.07
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        assert 'pred_spans' in outputs

        start_spans = targets['timestamp'].float()
        pred_spans = outputs['pred_spans'].float()
        src_spans = start_spans + pred_spans
        gt_spans = targets['span_label_nn'].float()
        mask =  targets['timestamp_mask'].bool()
        mask_full = targets['timestamp_mask'].unsqueeze(2).repeat(1, 1, 2)
        mask_valid =  targets['timestamp_window'].bool()
        mask_valid_full = targets['timestamp_window'].unsqueeze(2).repeat(1, 1, 2)
        loss_span = F.smooth_l1_loss(src_spans, gt_spans, reduction='none') * mask_valid_full
        loss_giou = 1 - torch.diag(generalized_temporal_iou(src_spans[mask_valid], gt_spans[mask_valid]))

        losses = {}
        losses['loss_b'] = loss_span.sum() / mask_valid.sum()
        losses['loss_g'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        src_logits = outputs['pred_logits'].squeeze(-1)
        mask = targets['timestamp_mask'].bool()
        mask_valid = targets['timestamp_window'].bool()
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[mask_valid] = 1
        target_classes.float()

        weights = torch.zeros_like(target_classes).float()
        weights[mask] = self.empty_weight[1].float()
        weights[mask_valid] = self.empty_weight[0].float()

        loss_ce = F.binary_cross_entropy(src_logits.float(), target_classes.float(), weight=weights,  reduction="none") * mask
        return {"loss_f": loss_ce.sum() / mask.sum()}

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}
        saliency_scores = targets["saliency_scores"]
        if saliency_scores.sum() == 0:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}

        # * inter-vid mode
        vid_mem_proj = outputs["vid_mem_proj"]
        pos_indices = targets["saliency_pos_labels"][:,0].long()
        batch_indices = torch.arange(len(vid_mem_proj)).to(vid_mem_proj.device)

        vid_feats = vid_mem_proj[batch_indices, pos_indices]
        txt_feats = outputs["txt_mem_proj"].squeeze(1)
        sim = sim_matrix(vid_feats, txt_feats)

        i_logsm = F.log_softmax(sim / self.temperature, dim=1)
        j_logsm = F.log_softmax(sim.t() /self.temperature, dim=1)

        idiag = torch.diag(i_logsm)
        jdiag = torch.diag(j_logsm)
        loss_i = idiag.sum() / len(idiag)
        loss_j = jdiag.sum() / len(jdiag)

        loss_saliency_inter = - loss_i - loss_j

        mask = targets['timestamp_mask']
        selected_scores = saliency_scores[batch_indices, pos_indices].unsqueeze(-1)
        neg_indices_in = (saliency_scores < selected_scores)
        neg_indices_in[batch_indices, pos_indices] = True
        mask_invalid = neg_indices_in * mask.bool()

        sim_in = F.cosine_similarity(vid_mem_proj, txt_feats.unsqueeze(1), dim=-1)
        sim_in = sim_in + (mask_invalid + 1e-45).log()
        logsm_in_i = F.log_softmax(sim_in / self.temperature, dim=1)
        logsm_in_j = F.log_softmax(sim_in.t() / self.temperature, dim=1)

        pos_logsm_in_i = logsm_in_i[batch_indices, pos_indices]
        pos_logsm_in_j = logsm_in_j[pos_indices, batch_indices]
        loss_in_i = pos_logsm_in_i.sum() / len(pos_logsm_in_i)
        loss_in_j = pos_logsm_in_j.sum() / len(pos_logsm_in_j)

        loss_saliency_intra = - loss_in_i - loss_in_j

        return {"loss_s_inter": loss_saliency_inter, "loss_s_intra": loss_saliency_intra}

    def loss_saliency_cls(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}
        saliency_scores = targets["saliency_scores"]
        if saliency_scores.sum() == 0:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}

        vid_mem_proj = outputs["vid_mem_proj"]
        pos_indices = targets["saliency_pos_labels"][:,0].long()  # (N, #pairs)
        batch_indices = torch.arange(len(vid_mem_proj)).to(vid_mem_proj.device)

        vid_feats = vid_mem_proj[batch_indices, pos_indices]
        txt_feats = outputs["txt_mem_proj"].squeeze(1)
        sim = sim_matrix(vid_feats, txt_feats)

        i_logsm = F.log_softmax(sim / self.temperature, dim=1)
        j_logsm = F.log_softmax(sim.t() /self.temperature, dim=1)

        idiag = torch.diag(i_logsm)
        jdiag = torch.diag(j_logsm)
        loss_i = idiag.sum() / len(idiag)
        loss_j = jdiag.sum() / len(jdiag)

        loss_saliency_inter = - loss_i - loss_j

        if 'cls_idx' not in targets.keys():
            return {"loss_s_inter": loss_saliency_inter}

        cls_indices = targets['cls_idx'].bool()
        cls_feats = outputs["cls_mem_proj"].squeeze(1)
        sim_cls = sim_matrix(vid_feats, cls_feats)

        i_logsm_cls = F.log_softmax(sim_cls / self.temperature, dim=1)
        idiag_cls = i_logsm_cls[cls_indices]
        loss_cls_i = idiag_cls.sum() / len(idiag_cls)

        loss_saliency_intra = - loss_cls_i

        return {"loss_s_inter": loss_saliency_inter, "loss_s_intra": loss_saliency_intra}

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "saliency_cls": self.loss_saliency_cls,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, hl_only=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        indices = None
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))
        return losses


class VTimeLLMMetaModel:

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args):
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        if model_args.clip_path:
            self.vision_tower = build_vision_tower(model_args)

        if not hasattr(self, 'mm_projector'):
            projector_type = getattr(model_args, 'mm_projector_type', 'linear')

            if projector_type == 'linear':
                self.mm_projector = nn.Linear(768, self.config.hidden_size)
            elif re.match(r'^mlp(\d+)x_gelu$', projector_type):
                mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(768, self.config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
                self.mm_projector = nn.Sequential(*modules)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print("load mlp:", pretrain_mm_mlp_adapter)

    def initialize_Loc_Head_modules(self, model_args, is_eval=False):
        self.span_loss_type = getattr(model_args, 'span_loss_type', 'l1')
        if self.span_loss_type == "l1":
            span_pred_dim = 2

        self.loc_interaction_type = getattr(model_args, 'loc_interaction_type', 'simp_add')

        if self.loc_interaction_type == 'simp_add':
            det_hidden_dim = 512  
            det_dim = 512  
            
        elif self.loc_interaction_type == 'intact_add':
            det_hidden_dim = 512
            det_dim = 512 

            loc_interact = [nn.Linear(det_hidden_dim*2, det_hidden_dim*2),
                            nn.ReLU(inplace=True),
                            nn.Linear(det_hidden_dim*2, det_dim),
                            nn.Dropout(0.0),]
            self.loc_interact = nn.ModuleList([nn.Sequential(*loc_interact)])
            self.loc_interact.train()
            for param in self.loc_interact.parameters():
                param.requires_grad = True

        elif self.loc_interaction_type == 'simp-atten':
            det_hidden_dim = 512
            det_dim = 512  

        elif self.loc_interaction_type == 'self-atten':
            det_hidden_dim = 512
            det_dim = 512  

            self.loc_interact = Transformer(d_model=det_dim,
                                            dropout=0.0,
                                            droppath=0.1,
                                            nhead=8,
                                            dim_feedforward=det_dim,
                                            num_encoder_layers=1,
                                            normalize_before=False,
                                            return_intermediate_dec=True,
                                            )
            self.vid_position_embedding = build_position_encoding(hidden_dim=det_dim, position_embedding_type='sine')

            self.token_type_embeddings = nn.Embedding(2, det_dim)
            self.token_type_embeddings.apply(init_weights)

            self.loc_interact.train()
            for param in self.loc_interact.parameters():
                param.requires_grad = True

            self.token_type_embeddings.train()
            for param in self.token_type_embeddings.parameters():
                param.requires_grad = True

        elif self.loc_interaction_type == 'gating':
            pass

        in_dim = self.config.hidden_size
        text_fc_det = [                    
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, det_hidden_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs_det = nn.ModuleList([nn.Sequential(*text_fc_det)])
        self.text_hidden_fcs_det.train()
        for param in self.text_hidden_fcs_det.parameters():
            param.requires_grad = True

        self.span_embed_head = Conv(det_dim, det_dim, span_pred_dim, 3, kernel_size=3)
        self.class_embed_head = Conv(det_dim, det_dim, 1, 3, kernel_size=3)

        for param in self.span_embed_head.parameters():
            param.requires_grad = True
        for param in self.class_embed_head.parameters():
            param.requires_grad = True

        self.patch_filter_type = getattr(model_args, 'patch_filter_type', 'only_cls')
        self.Tree_based_patch_filter = PatchFilter(num_cls=100, num_patch=0, patch_filter_type=self.patch_filter_type)

        self.det_weight_dict = {'loss_f': 2, 'loss_b': 5, 'loss_g': 2}
        self.det_losses = ['labels', 'spans']
        self.det_criterion = DetCriterion(self.det_weight_dict, self.det_losses, span_loss_type=self.span_loss_type, eos_coef=0.1)


class VTimeLLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                if self.get_model().config.model_type == 'chatglm':
                    target_shape = past_key_values[-1][-1].shape[0] + 1
                else:
                    target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        if type(images) is list:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.get_model().mm_projector(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.get_model().mm_projector(images)  
        # print([image.shape for image in image_features])
        
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_frame_masks = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0: 
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]  
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))  
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_frame_masks = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])  
                cur_frame_masks.append(torch.full((cur_labels_noim[i].shape[0],), 0, device=cur_labels.device, dtype=cur_labels.dtype))
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    midd_frame_mask = torch.full((cur_image_features.shape[0],), 0, device=cur_labels.device, dtype=cur_labels.dtype)
                    midd_frame_mask[:100] = 1  
                    cur_frame_masks.append(midd_frame_mask)

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_frame_masks = torch.cat(cur_frame_masks)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_frame_masks.append(cur_frame_masks)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []  
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        new_frame_masks_padded = torch.zeros((batch_size, max_len), dtype=new_labels[0].dtype, device=new_labels[0].device)
        for i, (cur_new_embed, cur_new_labels, cur_frame_masks) in enumerate(zip(new_input_embeds, new_labels, new_frame_masks)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    new_frame_masks_padded[i, :cur_len] = cur_frame_masks
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        new_frame_masks = new_frame_masks_padded.to(dtype=_attention_mask.dtype)

        if self.get_model().config.model_type == 'chatglm':
            fake_input_ids = torch.full((new_input_embeds.shape[0], new_input_embeds.shape[1]), -10000, 
                                        dtype=new_input_embeds.dtype, device=new_input_embeds.device)
            attention_mask = attention_mask.to(torch.int8)
            new_input_embeds = new_input_embeds.transpose(0, 1).contiguous()
        else:
            fake_input_ids = None

        return fake_input_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_frame_masks
