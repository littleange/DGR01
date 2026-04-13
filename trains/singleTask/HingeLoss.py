import torch
import torch.nn as nn
import torch.nn.functional as F
class HingeLoss(nn.Module):
    def __init__(self, margin_factor=0.15, pos_threshold=0.5):
        
        super(HingeLoss, self).__init__()
        self.margin_factor = margin_factor
        self.pos_threshold = pos_threshold

    def forward(self, ids, feats):
        if ids.dim() == 1:
            ids = ids.unsqueeze(1)
        feats = F.normalize(feats, p=2, dim=1) 
        
        batch_size = feats.shape[0]
        sim_matrix = torch.matmul(feats, feats.t()) 
        label_diff = torch.abs(ids - ids.t()) 
        is_positive = label_diff < self.pos_threshold 
        is_negative = ~is_positive 
        mask_not_self = ~torch.eye(batch_size, dtype=torch.bool, device=feats.device)
        is_positive = is_positive & mask_not_self
        is_negative = is_negative & mask_not_self
        sim_pos_masked = sim_matrix.clone()
        sim_pos_masked[~is_positive] = 100.0 
        hardest_pos_sim, _ = sim_pos_masked.min(dim=1, keepdim=True) 

        sim_neg_masked = sim_matrix.clone()
        sim_neg_masked[~is_negative] = -100.0
        hardest_neg_sim, neg_indices = sim_neg_masked.max(dim=1, keepdim=True) 

        current_margins = torch.gather(label_diff, 1, neg_indices) * self.margin_factor
        loss = torch.clamp(hardest_neg_sim - hardest_pos_sim + current_margins, min=0.0)
        valid_anchors = (is_positive.sum(1) > 0) & (is_negative.sum(1) > 0)
        
        if valid_anchors.sum() > 0:
            return loss[valid_anchors].mean()
        else:
            return torch.tensor(0.0, device=feats.device, requires_grad=True)