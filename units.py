import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class SomnialUnit(nn.Module):
	def __init__(self, in_channels, k=10):
		super().__init__()
		self.M = deque(maxlen=k)
		self.generator = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, kernel_size=1),
			nn.Identity()
		)

	def reshape_memory_sample(self, x_s, x_t):
		if x_s.shape[0] != x_t.shape[0]:
			pad_size = x_t.shape[0] - x_s.shape[0]
			if pad_size > 0:
				x_s = torch.cat([x_s, torch.zeros((pad_size, *x_s.shape[1:]), device=x_t.device)], dim=0)
			else:
				x_s = x_s[:x_t.shape[0]]
		return x_s

	def modulator(self, x_s_hat, x_t):
		q = F.normalize(x_t, dim=1)
		k = F.normalize(x_s_hat, dim=1)
		return torch.sigmoid((q * k).sum(dim=1, keepdim=True))

	def forward(self, x_t):
		if self.training:
			self.M.append(x_t.detach())
			x_s = random.choice(self.M)
		else:
			x_s = x_t
		x_s = self.reshape_memory_sample(x_s, x_t)
		x_s_hat = self.generator(x_s)
		m = self.modulator(x_s_hat, x_t)
		return m * x_s_hat + (1 - m) * x_t