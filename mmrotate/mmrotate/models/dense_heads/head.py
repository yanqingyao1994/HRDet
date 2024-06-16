import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmrotate.registry import MODELS
from mmrotate.models.dense_heads.rotated_fcos_head import RotatedFCOSHead

from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.utils import unpack_gt_instances

class MaskHead(BaseModule):
    def __init__(self, num_ins=5, num_convs=4, in_channels=256, conv_out_channels=256, fusion_level=1):
        super().__init__()
        self.num_ins = num_ins
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.fusion_level = fusion_level

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(ConvModule(self.in_channels, self.in_channels, 1))

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else conv_out_channels
            self.convs.append(ConvModule(in_channels, conv_out_channels, 3, 1, 1))

        self.conv_mask = nn.Conv2d(conv_out_channels, 1, 3, 1, 1)
        self.mask_loss = CrossEntropyLoss(use_sigmoid=True)

    def forward(self, feats):
        x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])

        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(feat, x.shape[-2:], mode='bilinear', align_corners=True)
                x = x + self.lateral_convs[i](feat)

        for i in range(self.num_convs):
            x = self.convs[i](x)

        mask_preds = self.conv_mask(x)
        return mask_preds

    def loss(self, mask_preds: Tensor, labels: Tensor) -> Tensor:
        labels = F.interpolate(labels.float(), scale_factor=1/16, mode='nearest') / 255
        loss_semantic_seg = self.mask_loss(mask_preds, labels)
        return loss_semantic_seg

@MODELS.register_module()
class FCOSHeadAug(RotatedFCOSHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_head = MaskHead()

    def loss(self, x, batch_data_samples):
        outs, semantic_pred = self(x)
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs
        loss_inputs = outs + (batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        gt_semantic_segs = [data_sample.gt_sem_seg.sem_seg for data_sample in batch_data_samples]
        loss_mask = self.mask_head.loss(semantic_pred, torch.stack(gt_semantic_segs))
        losses['loss_semantic_seg'] = loss_mask
        return losses

    def predict(self, x, batch_data_samples, rescale = False):
        outs, semantic_pred = self(x)
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        predictions = self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def forward(self, x):
        semantic_pred = self.mask_head(x)
        feats, masks = [], []
        for i, data in enumerate(x):
            feats.append(data.sum(1, True))
            mask = F.interpolate(semantic_pred, data.shape[-2:], mode='bilinear', align_corners=True)
            masks.append(mask.detach().round().bool())

        for i in range(len(self.strides)-1):
            mm = torch.cat([torch.pixel_unshuffle(masks[i], 2), masks[i+1]], 1)
            ww = torch.cat([torch.pixel_unshuffle(feats[i], 2), feats[i+1]], 1)
            ww = torch.softmax(ww.masked_fill(~mm, float('-inf')), 1).masked_fill(~mm, 1)
            feats[i], feats[i+1] = torch.pixel_shuffle(ww[:, :4], 2), ww[:, 4:]
        
        x = [data*feat for data, feat in zip(x, feats)]
        return super().forward(x), semantic_pred
