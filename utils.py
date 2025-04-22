import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import local_binary_pattern
from torchvision import transforms

def seed_functions(seed):
	"""Seeds functions from numpy and torch."""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class FocalLoss(nn.Module):
	def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction

	def forward(self, inputs, targets):
		log_probs = F.log_softmax(inputs, dim=1)
		probs = torch.exp(log_probs)
		target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
		log_target_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

		loss = -self.alpha * (1 - target_probs) ** self.gamma * log_target_probs
		if self.reduction == "mean":
			return loss.mean()
		if self.reduction == "sum":
			return loss.sum()
		return loss