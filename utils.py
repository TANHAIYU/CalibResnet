import sys
import numpy, torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_network(args, use_gpu=True):
	""" return given network"""
	if args.net == 'resnet34':
		from models.resnet import resnet34
		net = resnet34()
	elif args.net == 'resnet50':
		from models.resnet import resnet50
		net = resnet50()
	else:
		print('the network name you have entered is not supported yet')
		sys.exit()

	# if use_gpu == True:
	# 	net = net.cuda()
	# else:
	# 	net = net()
	return net


def get_training_dataloader(batch_size=16, num_workers=2, shuffle=True):
	transform_train = transforms.Compose([
		# transforms.ToPILImage(),
		# transforms.RandomCrop(32, padding=4),
		transforms.ToTensor(),
		# transforms.Normalize(mean, std)
	])
	cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,transform=transform_train)
	training_loader = DataLoader(cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

	return training_loader


def get_test_dataloader(batch_size=16, num_workers=2, shuffle=True):
	""" return training dataloader
    Returns: cifar100_test_loader:torch dataloader object
    """
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		# transforms.Normalize(mean, std)
	])
	cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
	test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

	return test_loader


# def compute_mean_std(cifar100_dataset):
# 	"""Returns:a tuple contains mean, std value of entire dataset"""
# 	data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
# 	data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
# 	data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
# 	mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
# 	std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)
#
# 	return mean, std


class WarmUpLR(_LRScheduler):
	"""warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

	def __init__(self, optimizer, total_iters, last_epoch=-1):
		self.total_iters = total_iters
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		"""we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
		return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
