import random
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import deque
from torchvision.models import ResNet50_Weights, ResNet152_Weights, DenseNet201_Weights

# Small units
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

# CNN architectures: ResNets
class RN50(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.freezable = True
		self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
		self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class LBPPatchedRN50(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.resnet50(pretrained=True)
		old_conv = self.model.conv1
		self.model.conv1 = nn.Conv2d(
			in_channels=4,
			out_channels=old_conv.out_channels,
			kernel_size=old_conv.kernel_size,
			stride=old_conv.stride,
			padding=old_conv.padding,
			bias=old_conv.bias is not None
		)
		with torch.no_grad():
			self.model.conv1.weight[:, :3] = old_conv.weight
			self.model.conv1.weight[:, 3] = old_conv.weight[:, 0]
		self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

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

class LBPPatchedRN152(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
		old_conv = self.model.conv1
		self.model.conv1 = nn.Conv2d(
			in_channels=4,
			out_channels=old_conv.out_channels,
			kernel_size=old_conv.kernel_size,
			stride=old_conv.stride,
			padding=old_conv.padding,
			bias=old_conv.bias is not None
		)
		with torch.no_grad():
			self.model.conv1.weight[:, :3] = old_conv.weight
			self.model.conv1.weight[:, 3] = old_conv.weight[:, 0]
		self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class DN201(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.densenet201(weights=DenseNet201_Weights.DEFAULT)
		self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class LBPPatchedDN201(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.densenet201(weights=DenseNet201_Weights.DEFAULT)
		old_conv = self.model.features.conv0
		self.model.features.conv0 = nn.Conv2d(
			in_channels=4,
			out_channels=old_conv.out_channels,
			kernel_size=old_conv.kernel_size,
			stride=old_conv.stride,
			padding=old_conv.padding,
			bias=old_conv.bias is not None
		)
		with torch.no_grad():
			self.model.features.conv0.weight[:, :3] = old_conv.weight
			self.model.features.conv0.weight[:, 3] = old_conv.weight[:, 0]
		self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class IV3(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.inception_v3(pretrained=True, aux_logits=False)
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

class MNV2(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.mobilenet_v2(pretrained=True)
		self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

	def forward(self, x):
		return self.model(x)

class LBPPatchedMNV2(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.mobilenet_v2(pretrained=True)
		old_conv = self.model.features[0][0]
		self.model.features[0][0] = nn.Conv2d(
			in_channels=4,
			out_channels=old_conv.out_channels,
			kernel_size=old_conv.kernel_size,
			stride=old_conv.stride,
			padding=old_conv.padding,
			bias=old_conv.bias is not None
		)
		with torch.no_grad():
			self.model.features[0][0].weight[:, :3] = old_conv.weight
			self.model.features[0][0].weight[:, 3] = old_conv.weight[:, 0]
		self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class MNV3S(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.mobilenet_v3_small(pretrained=True)
		self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class MNV3L(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.mobilenet_v3_large(pretrained=True)
		self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class LBPPatchedMNV3L(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.mobilenet_v3_large(pretrained=True)
		old_conv = self.model.features[0][0]
		self.model.features[0][0] = nn.Conv2d(
			in_channels=4,
			out_channels=old_conv.out_channels,
			kernel_size=old_conv.kernel_size,
			stride=old_conv.stride,
			padding=old_conv.padding,
			bias=old_conv.bias is not None
		)
		with torch.no_grad():
			self.model.features[0][0].weight[:, :3] = old_conv.weight
			self.model.features[0][0].weight[:, 3] = old_conv.weight[:, 0]
		self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

	def forward(self, x):
		return self.model(x)

# Experiments
class EVA2L(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = timm.create_model("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", pretrained=True, num_classes=num_classes)

	def forward(self, x):
		return self.model(x)

class LBPPatchedEVA2L(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)
		old_proj = self.model.patch_embed.proj
		self.model.patch_embed.proj = nn.Conv2d(
			in_channels=4,
			out_channels=old_proj.out_channels,
			kernel_size=old_proj.kernel_size,
			stride=old_proj.stride,
			padding=old_proj.padding,
			bias=old_proj.bias is not None
		)
		with torch.no_grad():
			self.model.patch_embed.proj.weight[:, :3] = old_proj.weight
			self.model.patch_embed.proj.weight[:, 3] = old_proj.weight[:, 0]
		self.model.head = nn.Linear(self.model.head.in_features, num_classes)

	def forward(self, x):
		return self.model(x)