"""
    To load various datasets
"""
import os, gc
import copy
import json
import numpy as np
from PIL import Image
from hydra.utils import get_original_cwd
from sklearn.model_selection import KFold

# torch...
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
# --- FIX 1: Dùng bí danh (alias) để tránh xung đột tên file ---
import datasets as hf_datasets
import math
from torch.utils.data import Subset, ConcatDataset

# ------------------------------------------------------------------------------
#   Globals
# ------------------------------------------------------------------------------
_tiny_train = os.path.join('datasets', 'tiny-imagenet-200', 'train')
_tiny_valid = os.path.join('datasets', 'tiny-imagenet-200', 'val', 'images')
# _cifar100_root và _cifar10_root được định nghĩa trực tiếp trong hàm,
# _imagenet32 được xử lý bởi Hugging Face


# ------------------------------------------------------------------------------
#   Loaders (for training functionalities)
# ------------------------------------------------------------------------------
def load_dataset(dataset, nbatch, normalize, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        trainset, validset = _load_cifar10(normalize=normalize)

        # : make loaders
        train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=nbatch, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(validset, \
                batch_size=nbatch, shuffle=False, **kwargs)

    # CIFAR100 dataset
    elif 'cifar100' == dataset:
        trainset, validset = _load_cifar100(normalize=normalize)

        # : make loaders
        train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=nbatch, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(validset, \
                batch_size=nbatch, shuffle=False, **kwargs)
                
    # ImageNet32 dataset
    elif 'imagenet32' == dataset:
        trainset, validset = _load_imagenet32_hf(normalize=normalize)

        # : make loaders
        train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=nbatch, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(validset, \
                batch_size=nbatch, shuffle=False, **kwargs)

    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        trainset, validset = _load_tiny_imagenet(normalize=normalize)

        # : make loaders
        train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=nbatch, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(validset, \
                batch_size=nbatch, shuffle=False, **kwargs)


    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    return train_loader, valid_loader



# ------------------------------------------------------------------------------
#   Internal functions
# ------------------------------------------------------------------------------

def _load_cifar100(normalize=True):
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    
    root_data = get_original_cwd() + '/datasets/cifar100'
    # root_data = 'datasets/cifar100'
    print(f"Load data at : {root_data}")

    if normalize:
        trainset = datasets.CIFAR100(root=root_data,
                                  train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(cifar100_mean, cifar100_std),
                                  ]))
        validset = datasets.CIFAR100(root=root_data,
                                  train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(cifar100_mean, cifar100_std),
                                  ]))
    else:
        trainset = datasets.CIFAR100(root=root_data,
                                  train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                  ]))
        validset = datasets.CIFAR100(root=root_data,
                                  train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))
    return trainset, validset

# --- FIX 2: Cập nhật hàm helper ImageNet32 ---
def _load_imagenet32_hf_raw():
    """
    Tải ImageNet 32x32 từ Hugging Face và trả về dữ liệu thô
    dưới dạng (train_data, train_labels, valid_data, valid_labels)
    giống như .data và .targets của CIFAR.
    """
    print("Loading ImageNet 32x32 from Hugging Face... (This may take time)")
    
    # --- Áp dụng cách load split_name của bạn ---
    ds_train = hf_datasets.load_dataset("benjamin-paine/imagenet-1k-32x32", split="train")
    ds_valid = hf_datasets.load_dataset("benjamin-paine/imagenet-1k-32x32", split="validation")
    
    # Process Training Set
    print("Processing train split...")
    train_images_pil = ds_train['image']
    train_labels = list(ds_train['label'])
    # Chuyển list các PIL Image thành 1 array NumPy (N, H, W, C)
    train_data = np.stack([np.array(img) for img in train_images_pil])
    
    # Process Validation Set
    print("Processing validation split...")
    valid_images_pil = ds_valid['image']
    valid_labels = list(ds_valid['label'])
    valid_data = np.stack([np.array(img) for img in valid_images_pil])
    
    print("ImageNet 32x32 processing complete.")
    
    return train_data, train_labels, valid_data, valid_labels

def _load_imagenet32_hf(normalize=True):
    """
    Tạo PyTorch Dataset (NumpyDataset) từ dữ liệu ImageNet32 thô.
    Hàm này được gọi bởi `load_dataset`.
    """
    # Lấy dữ liệu thô
    train_data, train_labels, valid_data, valid_labels = _load_imagenet32_hf_raw()
    
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    
    if normalize:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # Sử dụng NumpyDataset (đã có sẵn trong file của bạn)
    trainset = NumpyDataset(train_data, train_labels, transform=train_transform)
    validset = NumpyDataset(valid_data, valid_labels, transform=valid_transform)
    
    return trainset, validset


def _load_cifar10(normalize=True):
    root_data = get_original_cwd() + '/datasets/cifar10'
    # root_data = 'datasets/cifar10'
    print(f"Load data at : {root_data}")

    if normalize:
        trainset = datasets.CIFAR10(root=root_data,
                                  train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010)),
                                  ]))
        validset = datasets.CIFAR10(root=root_data,
                                  train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010)),
                                  ]))
    else:
        trainset = datasets.CIFAR10(root=root_data,
                                  train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                  ]))
        validset = datasets.CIFAR10(root=root_data,
                                  train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))
    return trainset, validset


def _load_tiny_imagenet(normalize=True):
    if normalize:
        trainset = datasets.ImageFolder(_tiny_train,
                              transform=transforms.Compose([
                                  transforms.RandomCrop(64, padding=8),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4802, 0.4481, 0.3975),
                                                       (0.2302, 0.2265, 0.2262)),
                              ]))
        validset = datasets.ImageFolder(_tiny_valid,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4802, 0.4481, 0.3975),
                                                       (0.2302, 0.2265, 0.2262)),
                              ]))
    else:
        trainset = datasets.ImageFolder(_tiny_train,
                              transform=transforms.Compose([
                                  transforms.RandomCrop(64, padding=8),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                              ]))
        validset = datasets.ImageFolder(_tiny_valid,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                              ]))
    return trainset, validset



# ------------------------------------------------------------------------------
#   Numpy dataset wrapper
# ------------------------------------------------------------------------------
class NumpyDataset(torch.utils.data.Dataset):
    """
        Numpy dataset
    """
    def __init__(self, data, labels, transform=None):
        self.data   = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        # to return a PIL Image
        data = Image.fromarray(data)

        # do transform...
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return self.data.shape[0]



# ------------------------------------------------------------------------------
#   Loaders (for causing misclassification of a specific sample)
# ------------------------------------------------------------------------------
def load_dataset_w_asample(dataset, sindex, clabel, slabel, nbatch, normalize, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_cifar10(normalize=normalize)

        # : compose the clean loaders
        ctrain_loader = torch.utils.data.DataLoader(clean_train, \
                batch_size=nbatch, shuffle=True, **kwargs)

        cvalid_loader = torch.utils.data.DataLoader(clean_valid, \
                batch_size=nbatch, shuffle=False, **kwargs)

        # : extract a sample from the valid dataset
        sample_data = clean_valid.data[sindex:sindex+1]     # H x W x C
        sample_clbl = [clabel]                               # [9] if not slabel else [slabel-1]
        sample_slbl = [slabel]

        # : compose two datasets
        if normalize:
            clean_sample = NumpyDataset( \
                sample_data, sample_clbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
            target_sample = NumpyDataset( \
                sample_data, sample_slbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))

        else:
            clean_sample = NumpyDataset( \
                sample_data, sample_clbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))
            target_sample = NumpyDataset( \
                sample_data, sample_slbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        csample_loader = torch.utils.data.DataLoader(clean_sample, \
                batch_size=nbatch, shuffle=False, pin_memory=True)

        tsample_loader = torch.utils.data.DataLoader(target_sample, \
                batch_size=nbatch, shuffle=False, pin_memory=True)
        return ctrain_loader, cvalid_loader, csample_loader, tsample_loader

    # CIFAR100 dataset
    elif 'cifar100' == dataset:
        cifar100_mean = (0.5071, 0.4867, 0.4408)
        cifar100_std = (0.2675, 0.2565, 0.2761)
        
        # : load cleans
        clean_train, clean_valid = _load_cifar100(normalize=normalize)

        # : compose the clean loaders
        ctrain_loader = torch.utils.data.DataLoader(clean_train, \
                batch_size=nbatch, shuffle=True, **kwargs)

        cvalid_loader = torch.utils.data.DataLoader(clean_valid, \
                batch_size=nbatch, shuffle=False, **kwargs)

        # : extract a sample from the valid dataset
        sample_data = clean_valid.data[sindex:sindex+1]     # H x W x C
        sample_clbl = [clabel]                               # [9] if not slabel else [slabel-1]
        sample_slbl = [slabel]

        # : compose two datasets
        if normalize:
            clean_sample = NumpyDataset( \
                sample_data, sample_clbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]))
            target_sample = NumpyDataset( \
                sample_data, sample_slbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]))

        else:
            clean_sample = NumpyDataset( \
                sample_data, sample_clbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))
            target_sample = NumpyDataset( \
                sample_data, sample_slbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        csample_loader = torch.utils.data.DataLoader(clean_sample, \
                batch_size=nbatch, shuffle=False, pin_memory=True)

        tsample_loader = torch.utils.data.DataLoader(target_sample, \
                batch_size=nbatch, shuffle=False, pin_memory=True)
        return ctrain_loader, cvalid_loader, csample_loader, tsample_loader
    
    # ImageNet32 dataset
    elif 'imagenet32' == dataset:
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        
        # : load cleans (sử dụng _load_imagenet32_hf để tạo PyTorch dataset)
        clean_train, clean_valid = _load_imagenet32_hf(normalize=normalize)

        # : compose the clean loaders
        ctrain_loader = torch.utils.data.DataLoader(clean_train, \
                batch_size=nbatch, shuffle=True, **kwargs)

        cvalid_loader = torch.utils.data.DataLoader(clean_valid, \
                batch_size=nbatch, shuffle=False, **kwargs)

        # : extract a sample from the valid dataset
        # Lấy data thô từ .data (vì _load_imagenet32_hf dùng NumpyDataset)
        sample_data = clean_valid.data[sindex:sindex+1]     # H x W x C
        sample_clbl = [clabel]                               
        sample_slbl = [slabel]

        # : compose two datasets
        if normalize:
            clean_sample = NumpyDataset( \
                sample_data, sample_clbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize(imagenet_mean, imagenet_std),
                ]))
            target_sample = NumpyDataset( \
                sample_data, sample_slbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize(imagenet_mean, imagenet_std),
                ]))

        else:
            clean_sample = NumpyDataset( \
                sample_data, sample_clbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))
            target_sample = NumpyDataset( \
                sample_data, sample_slbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        csample_loader = torch.utils.data.DataLoader(clean_sample, \
                batch_size=nbatch, shuffle=False, pin_memory=True)

        tsample_loader = torch.utils.data.DataLoader(target_sample, \
                batch_size=nbatch, shuffle=False, pin_memory=True)
        return ctrain_loader, cvalid_loader, csample_loader, tsample_loader
    
    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        return

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))


# ------------------------------------------------------------------------------
#   Backdoor dataset wrapper
# ------------------------------------------------------------------------------
class BackdoorDataset(torch.utils.data.Dataset):
    """
        Backdoor dataset
    """
    def __init__(self, data, labels, bshape, blabel, transform=None, idx_poison=None):
        self.data   = data
        self.labels = labels
        self.bshape = bshape
        self.blabel = blabel
        self.transform = transform
        self.is_poison = np.zeros(len(data), dtype=bool)
        if idx_poison is not None : 
            self.is_poison[idx_poison] = True
        self.is_bration=idx_poison is not None

    def __getitem__(self, index):
        cdata, clabel = self.data[index], self.labels[index]
        if self.is_bration == False : 
            bdata, blabel = _blend_backdoor(np.copy(cdata), self.bshape), self.blabel
        else : 
            if self.is_poison[index] : 
                bdata, blabel = _blend_backdoor(np.copy(cdata), self.bshape), self.blabel
            else : 
                bdata, blabel = self.data[index], self.labels[index]
        

        # to return a PIL Image
        cdata = Image.fromarray(cdata)
        bdata = Image.fromarray(bdata)

        # do transform...
        if self.transform:
            cdata = self.transform(cdata)
            bdata = self.transform(bdata)
        return cdata, clabel, bdata, blabel

    def __len__(self):
        return self.data.shape[0]


class BackdoorImageFolder(torchvision.datasets.DatasetFolder):
    """
        Backdoor dataset
    """
    def __init__(self, samples, targets, classes, class_to_idx, bshape, blabel, transform=None):
        self.classes = classes
        self.class_to_idx = class_to_idx

        # set the default loader...
        self.loader = default_loader

        self.samples = samples
        self.targets = targets
        self.bshape = bshape
        self.blabel = blabel
        self.transform = transform

    def __getitem__(self, index):
        # load data
        cpath, ctarget = self.samples[index]
        csample = np.array( self.loader(cpath) )
        bsample, btarget = _blend_backdoor(np.copy(csample), self.bshape), self.blabel

        # to return a PIL Image
        csample = Image.fromarray(csample)
        bsample = Image.fromarray(bsample)

        # do transform...
        if self.transform:
            csample = self.transform(csample)
            bsample = self.transform(bsample)
        return csample, ctarget, bsample, btarget

    def __len__(self):
        return len(self.samples)


"""
    Those functions from the torchvision
"""
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def _create_poison_idx(dataset, label, bratio) : 
    num_dataset = len(dataset) * 2
    num_poison = int(num_dataset * bratio)

    idx_poison = list(range(num_poison))

    return idx_poison

# ------------------------------------------------------------------------------
#   Backdoor loaders
# ------------------------------------------------------------------------------
def _blend_backdoor(data, shape):
    # retrive the data-shape
    h, w, c = data.shape

    # sanity checks
    assert (c == 3), ('Error: unsupported data - {}'.format(data.shape))

    # sanity checks
    assert (h == w), ('Error: should be square data - {}'.format(data.shape))

    # blend backdoor on it
    if 'square' == shape:
        valmin, valmax = data.min(), data.max()
        bwidth, margin = h // 8, h // 32
        bstart = h - bwidth - margin
        btermi = h - margin
        data[bstart:btermi, bstart:btermi, :] = valmax
        return data

    else:
        assert False, ('Error: unsupported shape - {}'.format(shape))
    # done.

def _blend_backdoor_multi(data, shape):
    # retrive the data-shape
    n, h, w, c = data.shape

    # sanity checks
    assert (c == 3), ('Error: unsupported data - {}'.format(data.shape))

    # sanity checks
    assert (h == w), ('Error: should be square data - {}'.format(data.shape))

    # blend backdoor on it
    if 'square' == shape:
        valmin, valmax = data.min(), data.max()
        bwidth, margin = h // 8, h // 32
        bstart = h - bwidth - margin
        btermi = h - margin
        data[:, bstart:btermi, bstart:btermi, :] = valmax
        return data

    else:
        assert False, ('Error: unsupported shape - {}'.format(shape))
    # done.

def load_backdoor(dataset, bshape, blabel, nbatch, normalize, kwargs, bratio=0.5):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_cifar10(normalize=normalize)

        # : extract the original data
        clean_tdata  = np.copy(clean_train.data)        # H x W x C
        clean_tlabel = copy.deepcopy(clean_train.targets)

        clean_vdata  = np.copy(clean_valid.data)
        clean_vlabel = copy.deepcopy(clean_valid.targets)

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        test_data, test_label = clean_vdata, clean_vlabel
        train_data, train_label, valid_data, valid_label = _split_data(clean_tdata, clean_tlabel)
        idx_poison_train = _create_poison_idx(train_data, train_label, bratio)

        # : compose as datasets
        if normalize:
            train_set  = BackdoorDataset( \
                train_data, train_label, bshape, blabel,
                transform=transforms.Compose([ 
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]), 
                idx_poison=idx_poison_train
            )
            valid_set  = BackdoorDataset( \
                valid_data, valid_label, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
            
            test_set  = BackdoorDataset( \
                test_data, test_label, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
        else:
            train_set  = BackdoorDataset( \
                train_data, train_label, bshape, blabel,
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]), 
                idx_poison=idx_poison_train
            )
            valid_set  = BackdoorDataset( \
                valid_data, valid_label, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))
            test_set  = BackdoorDataset( \
                test_data, test_label, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        test_set = _remove_blabel(test_set, blabel)

        train_loader = torch.utils.data.DataLoader( \
            train_set, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
            valid_set, batch_size=nbatch, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=nbatch, shuffle=False, **kwargs)

        return train_loader, valid_loader, test_loader

    # CIFAR100 dataset
    elif 'cifar100' == dataset:
        cifar100_mean = (0.5071, 0.4867, 0.4408)
        cifar100_std = (0.2675, 0.2565, 0.2761)
        
        # : load cleans
        clean_train, clean_valid = _load_cifar100(normalize=normalize)

        # : extract the original data
        clean_tdata  = np.copy(clean_train.data)        # H x W x C
        clean_tlabel = copy.deepcopy(clean_train.targets)

        clean_vdata  = np.copy(clean_valid.data)
        clean_vlabel = copy.deepcopy(clean_valid.targets)

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        test_data, test_label = clean_vdata, clean_vlabel
        train_data, train_label, valid_data, valid_label = _split_data(clean_tdata, clean_tlabel)
        idx_poison_train = _create_poison_idx(train_data, train_label, bratio)

        # : compose as datasets
        if normalize:
            train_set  = BackdoorDataset( \
                train_data, train_label, bshape, blabel,
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]), 
                idx_poison=idx_poison_train
            )
            valid_set  = BackdoorDataset( \
                valid_data, valid_label, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]))
            test_set  = BackdoorDataset( \
                test_data, test_label, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]))
        else:
            train_set  = BackdoorDataset( \
                train_data, train_label, bshape, blabel,
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]), 
                idx_poison=idx_poison_train
            )
            valid_set  = BackdoorDataset( \
                valid_data, valid_label, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))
            test_set  = BackdoorDataset( \
                test_data, test_label, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders

        test_set = _remove_blabel(test_set, blabel)

        train_loader = torch.utils.data.DataLoader( \
            train_set, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
            valid_set, batch_size=nbatch, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=nbatch, shuffle=False, **kwargs)

        return train_loader, valid_loader, test_loader

    # ImageNet32 dataset
    elif 'imagenet32' == dataset:
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        
        # : load cleans (Lấy data thô, không phải PyTorch dataset)
        clean_tdata, clean_tlabel, clean_vdata, clean_vlabel = _load_imagenet32_hf_raw()
        # Không cần del vì chúng ta không tạo clean_train

        # : compose as datasets
        if normalize:
            bdoor_train  = BackdoorDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(imagenet_mean, imagenet_std),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize(imagenet_mean, imagenet_std),
                ]))
        else:
            bdoor_train  = BackdoorDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader
        
    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_tiny_imagenet(normalize=normalize)

        # : extract the information
        clean_tclasses = clean_train.classes
        clean_tcls2idx = clean_train.class_to_idx
        clean_tsamples = clean_train.samples
        clean_ttargets = clean_train.targets

        clean_vclasses = clean_valid.classes
        clean_vcls2idx = clean_valid.class_to_idx
        clean_vsamples = clean_valid.samples
        clean_vtargets = clean_valid.targets

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : compose as datasets
        if normalize:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
        else:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    # done.

class SequentialKFold:
    def __init__(self, dataset, n_splits=5):
        self.dataset = dataset
        self.n_splits = n_splits
        self.N = len(dataset)
        self.fold_size = self.N // n_splits

    def split(self, fold_id):
        """
        fold_id: int ∈ [0, n_splits-1]
        """
        assert 0 <= fold_id < self.n_splits

        start = fold_id * self.fold_size
        end = (fold_id + 1) * self.fold_size if fold_id < self.n_splits - 1 else self.N

        valid_set = Subset(self.dataset, range(start, end))

        train_sets = []
        if start > 0:
            train_sets.append(Subset(self.dataset, range(0, start)))
        if end < self.N:
            train_sets.append(Subset(self.dataset, range(end, self.N)))

        train_set = ConcatDataset(train_sets)

        return train_set, valid_set
    

def split_dataset_by_fold(dataset, numrun, n_folds=5, seed=42):
    """
    dataset : PyTorch Dataset
    numrun  : fold index (0 → n_folds-1)
    """

    kf = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )

    splits = list(kf.split(dataset))
    train_idx, val_idx = splits[numrun]

    return train_idx, val_idx

def _remove_blabel(dataset, blabel) : 
    keep_indices = [
        i for i in range(len(dataset))
        if dataset[i][1] != blabel
    ]
    dataset = Subset(dataset, keep_indices)

    return dataset

def _split_data(clean_tdata, clean_tlabel): 
    
    train_idx, val_idx = split_dataset_by_fold(clean_tdata, 4)

    clean_tdata  = np.asarray(clean_tdata)
    clean_tlabel = np.asarray(clean_tlabel)

    train_data  = clean_tdata[train_idx]
    train_label = clean_tlabel[train_idx]

    valid_data  = clean_tdata[val_idx]
    valid_label = clean_tlabel[val_idx]

    return train_data, train_label, valid_data, valid_label

def blend_backdoor(dataset, bshape, blabel, bratio, nbatch, normalize, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_cifar10(normalize=normalize)

        # : extract the original data
        clean_tdata  = np.copy(clean_train.data)        # H x W x C
        clean_tlabel = copy.deepcopy(clean_train.targets)

        clean_vdata  = np.copy(clean_valid.data)
        clean_vlabel = copy.deepcopy(clean_valid.targets)

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : choose the base samples for crafting poisons
        num_trains = clean_tdata.shape[0]
        num_sample = int(num_trains * bratio)
        bdr_indexs = np.random.choice(num_trains, num_sample, replace=False)

        # : blend the backdoor (into the training data)
        bdoor_tdata  = _blend_backdoor_multi(clean_tdata[bdr_indexs], bshape)
        bdoor_tdata  = np.concatenate((clean_tdata, bdoor_tdata), axis=0)
        bdoor_tlabel = [blabel] * num_sample
        bdoor_tlabel = clean_tlabel + bdoor_tlabel

        # : compose as datasets
        if normalize:
            bdoor_train  = NumpyDataset( \
                bdoor_tdata, bdoor_tlabel,
                transform=transforms.Compose([ \
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
        else:
            bdoor_train  = NumpyDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader

    # CIFAR100 dataset
    elif 'cifar100' == dataset:
        cifar100_mean = (0.5071, 0.4867, 0.4408)
        cifar100_std = (0.2675, 0.2565, 0.2761)
        
        # : load cleans
        clean_train, clean_valid = _load_cifar100(normalize=normalize)

        # : extract the original data
        clean_tdata  = np.copy(clean_train.data)        # H x W x C
        clean_tlabel = copy.deepcopy(clean_train.targets)

        clean_vdata  = np.copy(clean_valid.data)
        clean_vlabel = copy.deepcopy(clean_valid.targets)

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : choose the base samples for crafting poisons
        num_trains = clean_tdata.shape[0]
        num_sample = int(num_trains * bratio)
        bdr_indexs = np.random.choice(num_trains, num_sample, replace=False)

        # : blend the backdoor (into the training data)
        bdoor_tdata  = _blend_backdoor_multi(clean_tdata[bdr_indexs], bshape)
        bdoor_tdata  = np.concatenate((clean_tdata, bdoor_tdata), axis=0)
        bdoor_tlabel = [blabel] * num_sample
        bdoor_tlabel = clean_tlabel + bdoor_tlabel

        # : compose as datasets
        if normalize:
            bdoor_train  = NumpyDataset( \
                bdoor_tdata, bdoor_tlabel,
                transform=transforms.Compose([ \
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]))
        else:
            bdoor_train  = NumpyDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader
        
    # ImageNet32 dataset
    elif 'imagenet32' == dataset:
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        
        # : load cleans (Lấy data thô)
        clean_tdata, clean_tlabel, clean_vdata, clean_vlabel = _load_imagenet32_hf_raw()

        # : choose the base samples for crafting poisons
        num_trains = clean_tdata.shape[0]
        num_sample = int(num_trains * bratio)
        bdr_indexs = np.random.choice(num_trains, num_sample, replace=False)

        # : blend the backdoor (into the training data)
        bdoor_tdata  = _blend_backdoor_multi(clean_tdata[bdr_indexs], bshape)
        bdoor_tdata  = np.concatenate((clean_tdata, bdoor_tdata), axis=0)
        bdoor_tlabel = [blabel] * num_sample
        bdoor_tlabel = clean_tlabel + bdoor_tlabel

        # : compose as datasets
        if normalize:
            bdoor_train  = NumpyDataset( \
                bdoor_tdata, bdoor_tlabel,
                transform=transforms.Compose([ \
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(imagenet_mean, imagenet_std),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize(imagenet_mean, imagenet_std),
                ]))
        else:
            bdoor_train  = NumpyDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader


    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_tiny_imagenet(normalize=normalize)

        # : extract the information
        clean_tclasses = clean_train.classes
        clean_tcls2idx = clean_train.class_to_idx
        clean_tsamples = clean_train.samples
        clean_ttargets = clean_train.targets

        clean_vclasses = clean_valid.classes
        clean_vcls2idx = clean_valid.class_to_idx
        clean_vsamples = clean_valid.samples
        clean_vtargets = clean_valid.targets

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : compose as datasets
        if normalize:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
        else:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    # done.

# ------------------------------------------------------------------------------
#   Testing module
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # This block only runs when you execute: python utils/datasets.py
    # It's for testing the functions in this file.
    
    import numpy as np 
    
    print("--- Running datasets.py in isolation for testing ---")

    # --- 1. Define Test Parameters ---
    
    # Danh sách các dataset cần test
    datasets_to_test = ['cifar10', 'cifar100', 'imagenet32']
    
    test_batch_size = 8  # Lấy 8 cái (gần 10) để kiểm tra
    use_norm = True
    dummy_shape = 'square'
    dummy_label = 0
    kwargs = {'num_workers': 0, 'pin_memory': False}
    
    # --- 2. Vòng lặp qua từng dataset ---
    for dataset_name in datasets_to_test:
        
        print("\n" + "="*60)
        print(f"--- TESTING DATASET: {dataset_name} ---")
        if dataset_name == 'imagenet32':
            print("This might take a while if downloading...")

        # --- 3. Call the Main Function (load_backdoor) ---
        try:
            train_loader, valid_loader = load_backdoor(
                dataset=dataset_name,
                bshape=dummy_shape,
                blabel=dummy_label,
                nbatch=test_batch_size,
                normalize=use_norm,
                kwargs=kwargs
            )
            
            print(f"\n[SUCCESS] {dataset_name} loaded.")
            
            # --- 4. Test the Loaders ---
            print("Fetching one batch from TRAIN loader...")
            c_images, c_labels, b_images, b_labels = next(iter(train_loader))
            
            print(f"  -> Clean Image batch shape: {c_images.shape} | Type: {c_images.dtype}")
            print(f"  -> Backdoor Image batch shape: {b_images.shape} | Type: {b_images.dtype}")
            print(f"  -> Clean Labels (first 3):   {c_labels[0:3].tolist()}")
            print(f"  -> Backdoor Labels (first 3): {b_labels[0:3].tolist()}")
            
            print("\nFetching one batch from VALID loader...")
            c_images_v, c_labels_v, b_images_v, b_labels_v = next(iter(valid_loader))
            print(f"  -> Clean Image batch shape: {c_images_v.shape} | Type: {c_images_v.dtype}")
            print(f"  -> Clean Labels (first 3):   {c_labels_v[0:3].tolist()}")
            
            print(f"\n--- Test for {dataset_name} complete. ---")

        except Exception as e:
            print(f"\n[FAILED] Error during {dataset_name} test:")
            print(f"Error: {e}")
            print("="*60)
            raise e # Tạm thời raise lỗi để gỡ bug nếu có
            
    print("\n" + "="*60)
    print("--- ALL DATASET TESTS FINISHED ---")