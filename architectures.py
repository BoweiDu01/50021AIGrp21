import random
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import deque
from torchvision.models import ResNet50_Weights, ResNet152_Weights, MobileNet_V2_Weights, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

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

class SMNV2(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		base = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
		self.stem = nn.Sequential(
			base.features[0],  # Conv-BN-ReLU
			SomnialUnit(in_channels=32, k=2)
		)
		self.features = base.features[1:]
		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(base.last_channel, num_classes)
		)

	def forward(self, x):
		x = self.stem(x)
		x = self.features(x)
		x = x.mean([2, 3])
		return self.classifier(x)

class MNV2(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
		self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

	def forward(self, x):
		return self.model(x)

class RNRS50(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = timm.create_model("resnetrs50", pretrained=True)
		self.model.reset_classifier(num_classes)

	def forward(self, x):
		return self.model(x)

class RN152(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
		self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class IV4(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = timm.create_model("inception_v4", pretrained=True, num_classes=num_classes)

	def forward(self, x):
		return self.model(x)