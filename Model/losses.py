import torch
import torch.nn as nn
import torch.nn.functional as F

# Weight BCE loss function giving higher weight to more rare substructures
class WeightedBCELoss(nn.Module):
    def __init__(self, frequencies, device, scaling_factor=1.0):
        super(WeightedBCELoss, self).__init__()
        # Convert frequencies to weights (higher weight for rare classes)
        epsilon = 1e-6
        weights = scaling_factor / (frequencies + epsilon)
        weights = weights * len(weights) / weights.sum()
        self.weights = torch.FloatTensor(weights).to(device)
        
    def forward(self, inputs, targets):
        loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        weighted_loss = loss * self.weights.unsqueeze(0)
        return weighted_loss.mean()