"""
Loss functions for handling extreme class imbalance in GoEmotions:
1. Standard CrossEntropyLoss
2. Class-Weighted CrossEntropyLoss
3. Multi-Class Focal Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_counts: list[int] | torch.Tensor, beta: float = 0.5):
        super().__init__()
        if not isinstance(class_counts, torch.Tensor):
            class_counts = torch.tensor(class_counts, dtype=torch.float)
        
        # Smoothed inverse frequency weighting
        # w_c = (N / (K * N_c))^beta
        total_samples = class_counts.sum()
        num_classes = len(class_counts)
        weights = (total_samples / (num_classes * class_counts.clamp(min=1))) ** beta
        # Normalize weights so mean is 1.0
        weights = weights / weights.mean()
        self.register_buffer('weights', weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights)

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | list[float] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [batch_size, num_classes], targets: [batch_size]
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        
        # Gather log_p and p for target classes
        target_log_p = log_p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        target_p = p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        
        # Modulating factor
        focal_weight = (1.0 - target_p) ** self.gamma
        
        loss = -focal_weight * target_log_p
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(dim=0, index=targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
