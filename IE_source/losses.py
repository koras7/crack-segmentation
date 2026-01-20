import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Flatten inputs and targets for segmentation
        if inputs.dim() > 2:
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, inputs.size(1))  # [B, C, H, W] → [N, C]
        else:
            inputs = inputs.reshape(-1, inputs.size(1))
    
        targets = targets.view(-1)  # [B, H, W] → [N]
    
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
    
        # Gather the probabilities of the true class
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
    
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            logpt = logpt * at
    
        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Assumes inputs are logits of shape [B, C, H, W]
        # Convert to probabilities
        inputs = F.softmax(inputs, dim=1)

        # One-hot encode targets [B, H, W] → [B, C, H, W]
        num_classes = inputs.size(1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        inputs = inputs.contiguous().view(inputs.size(0), inputs.size(1), -1)
        targets_onehot = targets_onehot.contiguous().view(targets_onehot.size(0), targets_onehot.size(1), -1)

        intersection = (inputs * targets_onehot).sum(dim=2)
        union = inputs.sum(dim=2) + targets_onehot.sum(dim=2)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        return loss

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', dice_weight=1.0, focal_weight=1.0):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        focal = self.focal(inputs, targets)
        dice = self.dice(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

# class BCEDiceLoss(nn.Module):
#     def __init__(self, class_weights=None,alpha=0.35, smooth=1e-5):
#         super(BCEDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.class_weights = class_weights
#         self.alpha = alpha
        
#         if class_weights is not None:
#             self.bce = nn.CrossEntropyLoss(weight=class_weights)
#         else:
#             self.bce = nn.CrossEntropyLoss()

#     def forward(self, inputs, targets):
#         bce_loss = self.bce(inputs, targets)

#         inputs_softmax = F.softmax(inputs, dim=1)
#         num_classes = inputs.size(1)
#         targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

#         inputs_flat = inputs_softmax.contiguous().view(inputs.size(0), num_classes, -1)
#         targets_flat = targets_onehot.contiguous().view(targets.size(0), num_classes, -1)

#         intersection = (inputs_flat * targets_flat).sum(dim=2)
#         union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)

#         dice = (2. * intersection + self.smooth) / (union + self.smooth)
#         dice_loss = 1 - dice.mean()

#         return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

# class BCEDiceLoss(nn.Module):
#     def __init__(self, class_weights=None, alpha=0.5, smooth=1e-5):
#         super(BCEDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.class_weights = class_weights
#         self.alpha = alpha
        
#         if class_weights is not None:
#             self.bce = nn.CrossEntropyLoss(weight=class_weights)
#         else:
#             self.bce = nn.CrossEntropyLoss()

#     def forward(self, inputs, targets):
#         bce_loss = self.bce(inputs, targets)
#         inputs_softmax = F.softmax(inputs, dim=1)
#         num_classes = inputs.size(1)

#         targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
#         inputs_flat = inputs_softmax.contiguous().view(inputs.size(0), num_classes, -1)
#         targets_flat = targets_onehot.contiguous().view(inputs.size(0), num_classes, -1)

#         intersection = (inputs_flat * targets_flat).sum(dim=2)
#         union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)
#         dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)

#         if self.class_weights is not None:
#             weights = self.class_weights.to(inputs.device)
#             weighted_dice = (weights * (1 - dice_per_class)).sum() / weights.sum()
#         else:
#             weighted_dice = (1 - dice_per_class).mean()

#         return self.alpha * bce_loss + (1 - self.alpha) * weighted_dice



class BCEDiceLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.alpha = alpha

        if class_weights is not None:
            self.bce = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.bce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = self.bce(inputs, targets)

        # Use only class 1 (crack) for Dice
        inputs_prob = F.softmax(inputs, dim=1)[:, 1:2, :, :]          # [B, 1, H, W]
        targets_bin = (targets == 1).float().unsqueeze(1)             # [B, 1, H, W]

        # Flatten
        inputs_flat = inputs_prob.contiguous().view(inputs.size(0), -1)
        targets_flat = targets_bin.contiguous().view(targets.size(0), -1)

        # Dice loss
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        # Final combined loss
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=0.5, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.class_weights = class_weights

        # BCE with optional class weights
        if class_weights is not None:
            self.bce = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.bce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        # --- BCE ---
        bce_loss = self.bce(inputs, targets)

        # --- Dice (for crack class only) ---
        probs = F.softmax(inputs, dim=1)[:, 1, :, :]   # Crack class probability [B,H,W]
        targets_bin = (targets == 1).float()           # Binary mask [B,H,W]

        intersection = (probs * targets_bin).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets_bin.sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        # --- Combine ---
        total_loss = self.alpha * bce_loss + (1 - self.alpha) * dice_loss
        return total_loss

class BalancedCrackLoss(nn.Module):
    """
    Balanced loss that prevents "predict everything" problem
    """
    def __init__(self, pos_weight=50.0, dice_weight=2.0, smooth=1e-6):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 2, H, W] - logits
            targets: [B, H, W] - ground truth (0 or 1)
        """
        # Get crack logits and targets
        crack_logits = inputs[:, 1, :, :]  # [B, H, W]
        crack_targets = (targets == 1).float()
        
        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(
            crack_logits,
            crack_targets,
            pos_weight=torch.tensor([self.pos_weight]).to(inputs.device)
        )
        
        # Dice Loss (for F1 optimization)
        crack_probs = torch.sigmoid(crack_logits)
        intersection = (crack_probs * crack_targets).sum()
        union = crack_probs.sum() + crack_targets.sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Combined
        return bce + self.dice_weight * dice_loss

