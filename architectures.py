import random
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import deque
from units import *

class SR50ViTB16(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
		self.backbone = timm.create_model("vit_base_r50_s16_224", pretrained=True, num_classes=0)
		self.backbone_stem = self.backbone.patch_embed.backbone
		in_channels = self.backbone_stem.feature_info[-1]['num_chs']
		self.somnial = SomnialUnit(in_channels)
		self.pool = nn.Identity()
		self.classifier = nn.Linear(self.backbone.num_features, 2)

	def forward(self, x):
		x = self.backbone_stem(x)
		x = self.somnial(x)
		x = self.backbone.patch_embed.proj(x)
		x = x.flatten(2).transpose(1, 2)
		if hasattr(self.backbone, 'pos_drop'):
			x = self.backbone.pos_drop(x + self.backbone.pos_embed[:, :x.size(1)])
		for blk in self.backbone.blocks:
			x = blk(x)
		if hasattr(self.backbone, 'norm'):
			x = self.backbone.norm(x)
		x = x[:, 0]
		x = self.pool(x)
		x = self.classifier(x)
		return x

class R50ViTB16(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
		self.backbone = timm.create_model("vit_base_r50_s16_224", pretrained=True, num_classes=0)
		self.pool = nn.Identity()
		self.classifier = nn.Linear(self.backbone.num_features, 2)

	def forward(self, x):
		x = self.backbone(x)
		x = self.pool(x)
		x = self.classifier(x)
		return x

class ViTB16(nn.Module):
	def __init__(self, num_classes=2):
		super(ViTB16, self).__init__()
		self.id = self.__class__.__name__
		self.model = models.vit_b_16(pretrained=True)
		self.model.heads = nn.Sequential(
			nn.Linear(self.model.hidden_dim, num_classes)
		)

	def forward(self, x):
		return self.model(x)

class RN50(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.resnet50(pretrained=True)
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
		self.model = models.resnet152(pretrained=True)
		self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class LBPPatchedRN152(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.resnet152(pretrained=True)
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

class DN121(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.densenet121(pretrained=True)
		self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class LBPPatchedDN121(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.densenet121(pretrained=True)
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

class DN169(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.densenet169(pretrained=True)
		self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class LBPPatchedDN169(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.densenet169(pretrained=True)
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

class DN201(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.densenet201(pretrained=True)
		self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class LBPPatchedDN201(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.densenet201(pretrained=True)
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

class Inceptionv3(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.Inception3(pretrained=True)
		self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

	def forward(self, x):
		return self.model(x)
	
class Densenet169(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.densenet169(pretrained=True)
		self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

	def forward(self, x):
		return self.model(x)	