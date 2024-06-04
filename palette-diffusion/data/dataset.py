import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import os
import torch
import numpy as np
from skimage.transform import resize
from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)
import glob
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class SpiralDepthTranslation(data.Dataset):
    def __init__(self, data_y, repeat=False):
        
        if repeat:
            self.y = np.load(data_y)[:1000, :, :, 1:]
            self.y = self.y[np.random.choice(len(data_y), 10)]
            self.y = np.repeat(self.y, 10, axis=0)
            self.y = np.transpose(self.y, [3,0,1,2])
            self.y = np.reshape(self.y, (-1, 128, 128, 1))

            self.x = np.load(data_y)[:1000, :, :, :-1]
            self.x = self.x[np.random.choice(len(data_y), 10)]
            self.x = np.repeat(self.x, 10, axis=0)
            self.x = np.transpose(self.x, [3,0,1,2])
            self.x = np.reshape(self.x, (-1, 128, 128, 1))
        else:
            self.y = np.load(data_y)[:50, :, :, 1:]
            self.y = np.transpose(self.y, [3,0,1,2])
            self.y = np.reshape(self.y, (-1, 128, 128, 1))

            self.x = np.load(data_y)[:50, :, :, :-1]
            self.x = np.transpose(self.x, [3,0,1,2])
            self.x = np.reshape(self.x, (-1, 128, 128, 1))
            
        self.tfs = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
        ])

    def __getitem__(self, index):
        ret = {}

        img = self.tfs((self.y[index] * 255).astype(np.uint8))
        cond_image = self.tfs((self.x[index] * 255).astype(np.uint8))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = f"{index%23}_to_{index%23 + 1}_{index//23}"
        return ret

    def __len__(self):
        return len(self.y)

class Spiral3D(data.Dataset):
    def __init__(self, data_x, data_y, image_size, d=16, repeat=False):
        self.x = np.load(data_x)[:, :, :, :5]
        self.y = np.load(data_y)[:, :, :, :d]
        if repeat:
            self.y = self.y[np.random.choice(len(data_y), 50)]
            self.y = np.repeat(self.y, 10, axis=0)

            self.x = self.x[np.random.choice(len(data_x), 50)]
            self.x = np.repeat(self.x, 10, axis=0)
            
        self.tfs_x = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5]*5, std=[.5]*5)
        ])
        self.tfs_y = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5]*d, std=[.5]*d)
        ])
        self.image_size = image_size
    def __getitem__(self, index):
        ret = {}
        y = resize(self.y[index] * 255, self.image_size)
        x = resize(self.x[index] * 255, self.image_size)

        img = self.tfs_y(y.astype(np.uint8))
        cond_image = self.tfs_x(x.astype(np.uint8))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = f"{index}"
        return ret

    def __len__(self):
        return len(self.y)

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[128, 128], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == "center_large":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//6, w//6, h//3*2, w//3*2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


class InpaintDatasetTemporal(data.Dataset):
    def __init__(self, data_path, test=False, repeat=False, mask_config={}, image_size=[128, 128]):
        imgs = np.load(data_path)
        # imgs = np.transpose(imgs, (0, 3, 1, 2))
        if not test:
            # imgs = imgs[:, :, :, 0::2]
            pass
        else:
            # imgs = imgs[:, :, :, 0::2]
            if not repeat:
                imgs = imgs[np.random.choice(len(imgs), 100)]
            else:
                imgs = imgs[np.random.choice(len(imgs), 10)]
                imgs = np.repeat(imgs, 10, axis=0)
        # imgs = imgs.reshape(-1, 128, 128, 5)
        self.imgs = imgs
        self.tfs_full = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5, .5, .5], std=[0.5, 0.5, 0.5, .5, .5])
        ])
        self.tfs_img = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        full_img = self.tfs_full((self.imgs[index] * 255).astype(np.uint8))
        pred_img = self.tfs_img((self.imgs[index][:, :, 0:1] * 255).astype(np.uint8))
        mask = self.get_mask()
        cond_image = full_img*(1. - mask) + mask*torch.randn_like(full_img)
        mask_img = pred_img*(1. - mask) + mask

        ret['gt_image'] = pred_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = f"{index}"
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == "center_large":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//6, w//6, h//3*2, w//3*2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)
 

class InpaintDatasetUV(data.Dataset):
    def __init__(self, data_path_u, data_path_v, test=False, repeat=False, mask_config={}, image_size=[128, 128]):
        imgs_u = np.load(data_path_u)
        imgs_v = np.load(data_path_v) / 2.3
        imgs = np.stack((imgs_u, imgs_v), axis=-1).reshape(imgs_u.shape[:-1] + (-1, ))
        # imgs = np.transpose(imgs, (0, 3, 1, 2))
        if not test:
            imgs = imgs[:18000]
        else:
            if not repeat:
                imgs = imgs[18000:]
                # imgs = imgs[np.random.choice(len(imgs), 10)]
            else:
                imgs = imgs[18000:]
                imgs = imgs[np.random.choice(len(imgs), 10)]
                imgs = np.repeat(imgs, 500, axis=0)
        # imgs = imgs.reshape(-1, 128, 128, 5)
        self.imgs = imgs
        self.tfs_full = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 10, std=[0.5] * 10)
        ])
        self.tfs_img = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
        ])
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        full_img = self.tfs_full((self.imgs[index] * 255).astype(np.uint8))
        pred_img = self.tfs_img((self.imgs[index][:, :, 0:2] * 255).astype(np.uint8))
        mask = self.get_mask()
        cond_image = full_img*(1. - mask) + mask*torch.randn_like(full_img)
        mask_img = pred_img*(1. - mask) + mask

        ret['gt_image'] = pred_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = f"{index}"
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == "center_large":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//6, w//6, h//3*2, w//3*2))
        elif self.mask_mode == "center_small":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h*3//8, w*3//8, h//4, w//4))
        elif self.mask_mode == "center_xlarge":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//8, w//8, h*3//4, w*3//4))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

class InpaintDatasetMaskRange(data.Dataset):
    def __init__(self, data_path, mask_config={}, image_size=[128, 128]):
        imgs = np.load(data_path)
        # imgs = np.transpose(imgs, (0, 3, 1, 2))
        #imgs = imgs[:, :, :, 0::2]
        imgs = imgs[np.random.choice(len(imgs), 20)]
        imgs = np.repeat(imgs, 10, axis=0)
        imgs = imgs.reshape(-1, 128, 128, 5)
        self.imgs = imgs
        self.tfs_full = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 5, std=[0.5] * 5)
        ])
        self.tfs_img = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.mask_range = np.arange(.05, .81, .05).astype(np.float32)
        self.image_size = image_size
        self.curr_mask_size = 0

    def __getitem__(self, index):
        ret = {}
        full_img = resize(self.imgs[index % len(self.imgs)] * 255, self.image_size)
        pred_img = resize(self.imgs[index % len(self.imgs)][:, :, 0:1] * 255, self.image_size)
        
        full_img = self.tfs_full(full_img.astype(np.uint8))
        pred_img = self.tfs_img(pred_img.astype(np.uint8))
        mask = self.get_mask()
        cond_image = full_img*(1. - mask) + mask*torch.randn_like(full_img)
        mask_img = pred_img*(1. - mask) + mask

        ret['gt_image'] = pred_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = f"{(index % len(self.imgs)) // 10}_{(index % len(self.imgs)) % 10}_{self.mask_range[self.curr_mask_size].round(2)}"
        self.curr_mask_size = index // len(self.imgs)

        return ret

    def __len__(self):
        return len(self.imgs) * len(self.mask_range)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mid = int((h * w * self.mask_range[self.curr_mask_size]) ** .5)
            mask = bbox2mask(self.image_size, (h//2 - mid//2, w//2 - mid//2, mid, mid))
        elif self.mask_mode == "center_large":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//6, w//6, h//3*2, w//3*2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

class NextTimeStep2D(data.Dataset):
    def __init__(self, data_folder, image_size=128, until=None, skip=None, num_prev=5):
        # self.x = np.load(data_x)[:, :, :, ::7]
        
        self.tfs_x = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5]*num_prev, std=[.5]*num_prev)
        ])

        self.tfs_y = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5]*5, std=[.5]*5)
        ])

        self.image_size = (image_size, image_size)
        self.num_prev = num_prev
        self.images = make_dataset(data_folder)
        self.images = sorted(self.images, key=lambda x: int(str(x)[20:-4]))
        if until is not None:
            self.images = self.images[:until]
        if skip is not None:
            self.images = self.images[skip:]



    def __getitem__(self, index):
        ret = {}
        # y = resize((self.y[index]/1.212)*255, self.image_size)
        x = []
        for i in range(self.num_prev):
            x.append(resize(np.array(pil_loader(self.images[i+index]))[:, :, 0:1], self.image_size))
        x = np.concatenate(x, axis=-1).astype(np.float32)

        y = []
        for i in range(5):
            y.append(resize(np.array(pil_loader(self.images[i+self.num_prev+index]))[:, :, 0:1], self.image_size))
        y = np.concatenate(y, axis=-1).astype(np.float32)
        assert 64 in y.shape
        # y = np.array(pil_loader(self.images[index+self.num_prev]))[:, :, 0:1]

        img = self.tfs_y(y)
        cond_image = self.tfs_x(x)

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = f"{index}"
        return ret

    def __len__(self):
        return len(self.images) - self.num_prev - 4

class NextTimeStep2DNPY(data.Dataset):
    def __init__(self, data_path, image_size=128, until=None, skip=None, num_prev=5):
        # self.x = np.load(data_x)[:, :, :, ::7]
        
        self.tfs_x = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5]*num_prev, std=[.5]*num_prev)
        ])

        self.tfs_y = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5]*5, std=[.5]*5)
        ])

        self.image_size = (image_size, image_size)
        self.num_prev = num_prev

        self.images = np.load(data_path)
        self.images = self.images.reshape(-1, self.images.shape[1], self.images.shape[2], 1)
        if until is not None:
            self.images = self.images[:until]
        if skip is not None:
            self.images = self.images[skip:]



    def __getitem__(self, index):
        ret = {}
        # y = resize((self.y[index]/1.212)*255, self.image_size)
        x = []
        for i in range(self.num_prev):
            x.append(resize(np.array(self.images[i+index])[:, :, 0:1], self.image_size))
        x = np.concatenate(x, axis=-1).astype(np.float32)

        y = []
        for i in range(5):
            y.append(resize(np.array(self.images[i+self.num_prev+index])[:, :, 0:1], self.image_size))
        y = np.concatenate(y, axis=-1).astype(np.float32)
        assert 64 in y.shape
        # y = np.array(pil_loader(self.images[index+self.num_prev]))[:, :, 0:1]

        img = self.tfs_y(y)
        cond_image = self.tfs_x(x)

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = f"{index}"
        return ret

    def __len__(self):
        return len(self.images) - self.num_prev - 4
    

class NextTimeStep2DEXP(data.Dataset):
    def __init__(self, data_path, image_size=128, until=None, skip=None, num_prev=5, num_after=5):
        # self.x = np.load(data_x)[:, :, :, ::7]
        
        self.tfs_x = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5]*num_prev, std=[.5]*num_prev)
        ])

        self.tfs_y = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5]*num_after, std=[.5]*num_after)
        ])

        self.image_size = (image_size, image_size)
        self.num_prev = num_prev
        self.num_after = num_after
        path = Path(data_path)
        self.images = []
        self.total_samples = 0
        # import ipdb; ipdb.set_trace()
        files = list(path.glob("*.npy"))
        if until is not None:
            files = files[:until]
        if skip is not None:
            files = files[skip:]

        for i in tqdm(files):
            d = resize(np.load(str(i))[:, :, :, 0].transpose(1, 2, 0), self.image_size).transpose(2, 0, 1)
            self.images.append(d)
            new = d[-min(len(d), 50):]

            self.images.append(ndimage.rotate(new[::3], 90, axes=(2, 1)))
            self.images.append(ndimage.rotate(new[1::3], 180, axes=(2, 1)))
            self.images.append(ndimage.rotate(new[2::3], 270, axes=(2, 1)))
            self.images.append(np.zeros((50, image_size, image_size)))

        
        for d in self.images:
            self.total_samples += d.shape[0] - self.num_prev - self.num_after + 1

    def item_map(self, index):
        curr = 0
        for idx in range(len(self.images)):
            if index < (len(self.images[idx]) - self.num_prev - self.num_after + 1 + curr):
                return index - curr, idx
            else:
                curr += len(self.images[idx]) - self.num_prev - self.num_after + 1
        
            

    def __getitem__(self, index):
        ret = {}
        # y = resize((self.y[index]/1.212)*255, self.image_size)
        start_index, img_index = self.item_map(index)
        imgs = self.images[img_index]
        x = []
        for i in range(self.num_prev):
            x.append(np.array(imgs[i+start_index]).reshape(*self.image_size, 1))
        x = np.concatenate(x, axis=-1).astype(np.float32)

        y = []
        for i in range(self.num_after):
            y.append(np.array(imgs[i+self.num_prev+start_index]).reshape(*self.image_size, 1))
        y = np.concatenate(y, axis=-1).astype(np.float32)

        img = self.tfs_y(y)
        cond_image = self.tfs_x(x)

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = f"{index}"
        return ret

    def __len__(self):
        return self.total_samples
    

class Autoregressive2DEXP(data.Dataset):
    def __init__(self, data_path, image_size=128, num_prev=5, num_after=5):
        # self.x = np.load(data_x)[:, :, :, ::7]
        
        self.tfs_x = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5]*num_prev, std=[.5]*num_prev)
        ])

        self.tfs_y = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5]*num_after, std=[.5]*num_after)
        ])

        self.image_size = (image_size, image_size)

        self.images = np.load(data_path)

    def __getitem__(self, index):
        ret = {}
        img = self.tfs_y(resize(self.images[index], self.image_size))
        cond_image = self.tfs_x(resize(self.images[index], self.image_size))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = f"{index}"
        return ret

    def __len__(self):
        return len(self.images)

class CondGeneration(data.Dataset):
    def __init__(self, data_files, image_size=128):
        # self.x = np.load(data_x)[:, :, :, ::7]
        self.tfs_y = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5]*10, std=[.5]*10)
        ])

        self.image_size = (image_size, image_size)
        self.labels = []
        self.images = []
        for file_suffix in data_files:
            file = "../data/parameter_npys/u_" + file_suffix + ".npy"
            file_v = "../data/parameter_npys/v_" + file_suffix + ".npy"

            u_dat = np.load(file)
            v_dat = np.load(file_v)
            dat = np.concatenate([u_dat, v_dat], axis=-1)[:500]

            lab = np.repeat([list(map(float, file_suffix.split("_")))], len(dat), axis=0)
            self.images.append(dat)
            self.labels.append(lab)

        self.images = np.concatenate(self.images)
        self.labels = np.concatenate(self.labels)

    def __getitem__(self, index):
        ret = {}

        img = self.tfs_y(resize(self.images[index], (64, 64)))
        
        cond = self.labels[index:index+1].repeat(64**2, axis=0).reshape(64, 64, 2).transpose([2, 0, 1])
        cond_image = torch.FloatTensor(cond)

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = f"{self.labels[index][0]}_{self.labels[index][1]}_{index}"
        return ret

    def __len__(self):
        return len(self.images)
        # return 100
    
class CondGenerationNew(data.Dataset):
    def __init__(self, data_files, data_prefix, param_1, param_2, image_size=128):
        # self.x = np.load(data_x)[:, :, :, ::7]
        self.tfs_y = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            v2.ToTensor(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.Normalize(mean=[.5]*10, std=[.5]*10)
        ])
        self.label_matrix = np.dstack(np.meshgrid(param_1, param_2))

        self.image_size = image_size
        self.labels = []
        self.images = []
        for file_suffix in tqdm(data_files):
            file = data_prefix + file_suffix
            y, x = list(map(lambda x: int(x) - 1, file_suffix.split("-")[:2]))
            dat = np.load(file)
            dat = dat[:, ::3].transpose([0, 2, 3, 1, 4]).reshape(len(dat), 128, 128, -1)
            dat[..., 1::2] = dat[..., 1::2] / 3.4
            
            dat = resize(dat, (len(dat), self.image_size, self.image_size))
            lab = np.repeat(self.label_matrix[x, y][None, :], len(dat), axis=0)
            self.images.append(dat[:-5])
            self.labels.append(lab[:-5])

        self.images = np.concatenate(self.images)
        self.labels = np.concatenate(self.labels)

    def __getitem__(self, index):
        ret = {}
        img = self.tfs_y(self.images[index])
        rotate = torch.randint(0, 4, ()).type(torch.FloatTensor)
        img = transforms.functional.rotate(img, (rotate*90).item())
        
        cond = self.labels[index:index+1].repeat(self.image_size**2, axis=0).reshape(self.image_size, self.image_size, 2).transpose([2, 0, 1])
        cond_image = torch.FloatTensor(cond)

        ret['gt_image'] = img.to(torch.float)
        ret['cond_image'] = cond_image
        ret['path'] = f"{self.labels[index][0]}_{self.labels[index][1]}_{index}"
        return ret

    def __len__(self):
        return len(self.images)
        # return 100


class CondGenerationCrossAttn(data.Dataset):
    def __init__(self, data_files, data_prefix, param_1, param_2, image_size=128):
        # self.x = np.load(data_x)[:, :, :, ::7]
        self.tfs_y = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            v2.ToTensor(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.Normalize(mean=[.5]*10, std=[.5]*10)
        ])
        self.label_matrix = -np.log(np.dstack(np.meshgrid(param_1, param_2)))
        # self.label_matrix = np.dstack(np.meshgrid(param_1, param_2))

        self.image_size = image_size
        self.labels = []
        self.images = []
        for file_suffix in tqdm(data_files):
            file = data_prefix + file_suffix
            y, x = list(map(lambda x: int(x) - 1, file_suffix.split("-")[:2]))
            dat = np.load(file)
            dat = dat[:, ::3].transpose([0, 2, 3, 1, 4]).reshape(len(dat), 128, 128, -1)
            dat[..., 1::2] = dat[..., 1::2] / 3.4
            
            dat = resize(dat, (len(dat), self.image_size, self.image_size))
            lab = np.repeat(self.label_matrix[x, y][None, :], len(dat), axis=0)
            self.images.append(dat)
            self.labels.append(lab)

        self.images = np.concatenate(self.images)
        self.labels = np.concatenate(self.labels)

    def __getitem__(self, index):
        ret = {}
        img = self.tfs_y(self.images[index])
        rotate = torch.randint(0, 4, ()).type(torch.FloatTensor)
        img = transforms.functional.rotate(img, (rotate*90).item())
        
        cond = self.labels[index]
        cond_image = torch.FloatTensor(cond)

        ret['gt_image'] = img.to(torch.float)
        ret['cond_image'] = cond_image
        ret['path'] = f"{self.labels[index][0]}_{self.labels[index][1]}_{index}"
        ret['cross'] = True
        
        return ret

    def __len__(self):
        return len(self.images)
        # return 100

class InpaintDatasetCustom(data.Dataset):
    def __init__(self, mask_config={}, image_size=[128, 128]):
        sim_imgs = np.load("../data/sim_test.npy")[400:401]
        # imgs = np.transpose(imgs, (0, 3, 1, 2))
        #imgs = imgs[:, :, :, 0::2]
        sim_imgs = np.repeat(sim_imgs, 500, axis=0)
        sim_imgs = sim_imgs.reshape(-1, 128, 128, 5)

        com_imgs = np.load("../data/com_test.npy")[200:201]
        # imgs = np.transpose(imgs, (0, 3, 1, 2))
        #imgs = imgs[:, :, :, 0::2]
        com_imgs = np.repeat(com_imgs, 500, axis=0)
        com_imgs = com_imgs.reshape(-1, 128, 128, 5)
        self.imgs = np.concatenate([sim_imgs, com_imgs], axis=0)
        self.tfs_full = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 5, std=[0.5] * 5)
        ])
        self.tfs_img = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.mask_range = [0.3, 0.7]
        self.image_size = image_size
        self.curr_mask_size = 0

    def __getitem__(self, index):
        ret = {}
        full_img = resize(self.imgs[index % len(self.imgs)] * 255, self.image_size)
        pred_img = resize(self.imgs[index % len(self.imgs)][:, :, 0:1] * 255, self.image_size)
        
        full_img = self.tfs_full(full_img.astype(np.uint8))
        pred_img = self.tfs_img(pred_img.astype(np.uint8))
        mask = self.get_mask()
        cond_image = full_img*(1. - mask) + mask*torch.randn_like(full_img)
        mask_img = pred_img*(1. - mask) + mask

        ret['gt_image'] = pred_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = f"{(index % len(self.imgs)) // 10}_{(index % len(self.imgs)) % 10}_{self.mask_range[self.curr_mask_size]}"
        self.curr_mask_size = index // len(self.imgs)

        return ret

    def __len__(self):
        return len(self.imgs) * len(self.mask_range)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mid = int((h * w * self.mask_range[self.curr_mask_size]) ** .5)
            mask = bbox2mask(self.image_size, (h//2 - mid//2, w//2 - mid//2, mid, mid))
        elif self.mask_mode == "center_large":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//6, w//6, h//3*2, w//3*2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)
    

class CellCultureDataset(data.Dataset):
    def __init__(self, image_size=128, history_len=32, skip=8, take=108, after=0, noise=0.4):
        self.u_imgs = []
        self.v_imgs = []
        self.positions = []
        self.history_len = history_len

        files = glob.glob("/mnt/data1/shared/data/cellculture-data/simulation/seed_*")
        print(len(files) // 3)
        files = sorted(files, key=lambda y: y[56:61])[after*3:take*3 + after*3]
        for file in files:
            if "_u.n" in file:
                self.u_imgs.append(np.load(file))
            elif "_v.n" in file:
                self.v_imgs.append(np.load(file))
            elif "_positions.n" in file:
                self.positions.append(np.load(file))

        self.k_i_map = {}
        self.index = 0
        for k in range(len(self.positions)):
            for i in range(0, len(self.positions[k])-history_len, skip):
                self.k_i_map[self.index] = (k, i)
                self.index += 1

        self.tfs_disp = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * history_len * 2, std=[0.5] * history_len * 2),
                transforms.Lambda(lambda tensor: (tensor + torch.randn_like(tensor) * noise))
        ])
        self.tfs_uv = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 2, std=[0.5] * 2)
        ])
        self.image_size = image_size

    def __getitem__(self, index):
        k, i = self.k_i_map[index]

        disp_dat = np.zeros((128, 128, self.history_len*2))
        for j in range (0, self.history_len):
            disp_dat[:, :, j*2:(j+1)*2] = (self.positions[k][i+j] - self.positions[k][i])
        uv_dat = np.dstack([self.u_imgs[k][i], self.v_imgs[k][i] / 2.3])

        ret = {}
        disp = resize(disp_dat, (self.image_size, self.image_size))
        uv = resize(uv_dat, (self.image_size, self.image_size))
        
        disp = self.tfs_disp(disp)
        uv = self.tfs_uv(uv)

        ret['gt_image'] = uv.type(torch.FloatTensor)
        ret['cond_image'] = disp.type(torch.FloatTensor)
        ret['path'] = f"{index}"

        return ret

    def __len__(self):
        return self.index