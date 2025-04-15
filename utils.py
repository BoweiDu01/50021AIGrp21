import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import torch
from skimage.feature import local_binary_pattern
from torchvision import transforms

def compute_lbp_tensor(img):
	"""
	Input: tensor of shape (3, H, W)
	Output: tensor of shape (1, H, W)
	"""
	gray = transforms.functional.rgb_to_grayscale(img)
	gray_np = gray.squeeze(0).numpy()
	lbp = local_binary_pattern(gray_np, P=8, R=1, method="uniform")
	lbp = torch.tensor(lbp, dtype=torch.float32) / lbp.max()
	return lbp.unsqueeze(0)

class LBPDataset(torch.utils.data.Dataset):
	def __init__(self, imagefolder_dataset):
		self.dataset = imagefolder_dataset

	def __getitem__(self, index):
		img, label = self.dataset[index]
		lbp = compute_lbp_tensor(img)
		img_with_lbp = torch.cat([img, lbp], dim=0)
		return img_with_lbp, label

	def __len__(self):
		return len(self.dataset)

def seed_functions(seed):
	"""Seeds functions from numpy and torch."""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def multivisualise_training_loss(models):
	plt.figure(figsize=(8, 5))
	for model in models:
		model_folder = os.path.join(MODELS_PATH, f"{model.id}")
		with open(os.path.join(model_folder, "losses.json"), "r") as f:
			loss_data = json.load(f)
		train_losses = loss_data["train_losses"]
		epochs = range(1, len(train_losses) + 1)
		plt.plot(epochs, train_losses, label=f"{model.id} - train")
	plt.xlabel("Epoch")
	plt.ylabel("Training Loss")
	plt.legend()
	plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.tight_layout()
	plt.show()

def multivisualise_validation_loss(models):
	plt.figure(figsize=(8, 5))
	for model in models:
		model_folder = os.path.join(MODELS_PATH, f"{model.id}")
		with open(os.path.join(model_folder, "losses.json"), "r") as f:
			loss_data = json.load(f)
		dev_losses = loss_data["dev_losses"]
		epochs = range(1, len(dev_losses) + 1)
		plt.plot(epochs, dev_losses, label=f"{model.id} - val")
	plt.xlabel("Epoch")
	plt.ylabel("Validation Loss")
	plt.legend()
	plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.tight_layout()
	plt.show()