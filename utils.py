import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import torch
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

class BinaryConfusionMatrix:
	def __init__(self, *args):
		if len(args) == 1 and isinstance(args[0], dict):
			d = args[0]
			self.tp = d["tp"]
			self.tn = d["tn"]
			self.fp = d["fp"]
			self.fn = d["fn"]
		elif len(args) == 4:
			self.tp, self.tn, self.fp, self.fn = args
		else:
			raise ValueError("Expected either (tp, tn, fp, fn) or a dict with keys 'tp', 'tn', 'fp', 'fn'.")

	def __str__(self):
		return (
			f"accuracy:    {self.accuracy():.8f}\n"
			f"precision:   {self.precision():.8f}\n"
			f"recall:      {self.recall():.8f}\n"
			f"binary_f1:   {self.binary_f1():.8f}\n"
			f"macro_f1:    {self.macro_f1():.8f}"
		)

	def accuracy(self):
		total = self.tp + self.tn + self.fp + self.fn
		if total == 0:
			return 0.0
		return round((self.tp + self.tn) / total, 8)

	def precision(self):
		denominator = self.tp + self.fp
		if denominator == 0:
			return 0.0
		return round(self.tp / denominator, 8)

	def recall(self):
		denominator = self.tp + self.fn
		if denominator == 0:
			return 0.0
		return round(self.tp / denominator, 8)

	def binary_f1(self):
		p = self.precision()
		r = self.recall()
		if p + r == 0:
			return 0.0
		return round(2 * p * r / (p + r), 8)

	def macro_f1(self):
		p_pos = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
		r_pos = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
		f1_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) else 0.0

		p_neg = self.tn / (self.tn + self.fn) if (self.tn + self.fn) else 0.0
		r_neg = self.tn / (self.tn + self.fp) if (self.tn + self.fp) else 0.0
		f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) else 0.0

		return round((f1_pos + f1_neg) / 2, 8)