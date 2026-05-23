from torchvision import transforms
from perlin import perlin_mask
from enum import Enum

import numpy as np
import pandas as pd
import cv2

import PIL
import torch
import os


class CLAHE(object):
    """CLAHE 전처리 transform (PIL Image → PIL Image)"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return PIL.Image.fromarray(result)
import glob

_CLASSNAMES = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
            self,
            source,
            anomaly_source_path='/root/dataset/dtd/images',
            dataset_name='mvtec',
            classname='leather',
            resize=288,
            resize_w=0,
            imagesize=288,
            imagesize_w=0,
            split=DatasetSplit.TRAIN,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=0,
            rand_aug=1,
            downsampling=8,
            scale=0,
            batch_size=8,
            inject_prob=1/32,
            **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
            inject_prob: [float]. Probability of injecting a real defect
                         image instead of Perlin synthesis during training.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.inject_prob = inject_prob
        self._h_flip_p = h_flip_p
        self._v_flip_p = v_flip_p
        self._gray_p = gray_p
        self._rotate_degrees = rotate_degrees
        self._translate = translate
        self._scale = scale
        self._clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
        self._color_jitter = transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor)
        self.batch_size = batch_size
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.fg = fg
        self.rand_aug = rand_aug
        self.downsampling = downsampling
        self.imgsize_h = imagesize
        self.imgsize_w = imagesize_w if imagesize_w > 0 else imagesize
        self.resize_h = resize
        self.resize_w = resize_w if resize_w > 0 else resize
        if self.distribution == 1:
            self.resize = [self.resize_h, self.resize_w]
        else:
            self.resize = (self.resize_h, self.resize_w)
        self.imgsize = (self.imgsize_h, self.imgsize_w)
        self.imagesize = (3, self.imgsize_h, self.imgsize_w)
        self.classname = classname
        self.dataset_name = dataset_name

        if self.distribution != 1 and (self.classname == 'toothbrush' or self.classname == 'wood'):
            ratio = 329 / 288
            self.resize = (round(self.imgsize_h * ratio), round(self.imgsize_w * ratio))

        xlsx_path = './datasets/excel/' + self.dataset_name + '_distribution.xlsx'
        if self.fg == 2:  # choose by file
            try:
                df = pd.read_excel(xlsx_path)
                self.class_fg = df.loc[df['Class'] == self.dataset_name + '_' + classname, 'Foreground'].values[0]
            except:
                self.class_fg = 1
        elif self.fg == 1:  # with foreground mask
            self.class_fg = 1
        else:  # without foreground mask
            self.class_fg = 0

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.anomaly_source_paths = sorted(1 * glob.glob(anomaly_source_path + "/*/*.jpg") +
                                           0 * list(next(iter(self.imgpaths_per_class.values())).values())[0])

        # 실제 결함 이미지 + 마스크 쌍 수집 (TRAIN 시 주입용)
        self.real_defect_pairs = []
        if self.split == DatasetSplit.TRAIN and self.inject_prob > 0:
            test_bad_dir = os.path.join(self.source, self.classname, "test", "bad")
            gt_bad_dir = os.path.join(self.source, self.classname, "ground_truth", "bad")
            if os.path.isdir(test_bad_dir) and os.path.isdir(gt_bad_dir):
                for fname in sorted(os.listdir(test_bad_dir)):
                    img_p = os.path.join(test_bad_dir, fname)
                    mask_p = os.path.join(gt_bad_dir, fname)
                    if os.path.isfile(mask_p):
                        self.real_defect_pairs.append((img_p, mask_p))

        self.transform_img = [
            transforms.Resize(self.resize),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.imgsize),  # (H, W) tuple
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),  # (H, W) tuple
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def rand_augmenter(self):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = [
            transforms.Resize(self.resize),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.imgsize),  # (H, W) tuple
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        transform_aug = transforms.Compose(transform_aug)
        return transform_aug

    def _transform_defect_pair(self, img_pil, mask_pil):
        """결함 이미지와 마스크에 동일한 공간 변환을 동기화하여 적용."""
        F = transforms.functional
        # Resize
        img = F.resize(img_pil, self.resize)
        mask = F.resize(mask_pil, self.resize,
                        interpolation=transforms.InterpolationMode.NEAREST)
        # CLAHE, ColorJitter (image only)
        img = self._clahe(img)
        img = self._color_jitter(img)
        # RandomHorizontalFlip (synced)
        if np.random.random() < self._h_flip_p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        # RandomVerticalFlip (synced)
        if np.random.random() < self._v_flip_p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        # RandomGrayscale (image only)
        if np.random.random() < self._gray_p:
            img = F.rgb_to_grayscale(img, num_output_channels=3)
        # RandomAffine (synced)
        img_size = [img.size[1], img.size[0]]
        angle, trans, scale, shear = transforms.RandomAffine.get_params(
            degrees=(-self._rotate_degrees, self._rotate_degrees),
            translate=(self._translate, self._translate) if self._translate else None,
            scale_ranges=(1.0 - self._scale, 1.0 + self._scale) if self._scale else None,
            shears=None, img_size=img_size)
        img = F.affine(img, angle, trans, scale, shear,
                       interpolation=transforms.InterpolationMode.BILINEAR)
        mask = F.affine(mask, angle, trans, scale, shear,
                        interpolation=transforms.InterpolationMode.NEAREST)
        # CenterCrop
        img = F.center_crop(img, self.imgsize)
        mask = F.center_crop(mask, self.imgsize)
        # ToTensor + Normalize
        img = F.normalize(F.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
        mask = F.to_tensor(mask)
        return img, mask

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        mask_fg = mask_s = aug_image = torch.tensor([1])
        if self.split == DatasetSplit.TRAIN:
            use_real = (self.real_defect_pairs
                        and np.random.random() < self.inject_prob)

            if use_real:
                # 실제 결함 이미지 + 바이너리 마스크 주입 (공간 변환 동기화)
                di = np.random.randint(len(self.real_defect_pairs))
                defect_img_path, defect_mask_path = self.real_defect_pairs[di]
                aug_image, defect_mask = self._transform_defect_pair(
                    PIL.Image.open(defect_img_path).convert("RGB"),
                    PIL.Image.open(defect_mask_path).convert("L"))
                feat_size = (self.imgsize_h // self.downsampling,
                             self.imgsize_w // self.downsampling)
                mask_s = torch.ceil(defect_mask[0])
                mask_s = torch.nn.functional.max_pool2d(
                    mask_s.unsqueeze(0).unsqueeze(0),
                    kernel_size=(self.imgsize_h // feat_size[0],
                                 self.imgsize_w // feat_size[1]),
                ).squeeze()
            else:
                # 기존 Perlin 합성
                aug = PIL.Image.open(np.random.choice(self.anomaly_source_paths)).convert("RGB")
                if self.rand_aug:
                    transform_aug = self.rand_augmenter()
                    aug = transform_aug(aug)
                else:
                    aug = self.transform_img(aug)

                if self.class_fg:
                    fgmask_path = image_path.split(classname)[0] + 'fg_mask/' + classname + '/' + os.path.split(image_path)[-1]
                    mask_fg = PIL.Image.open(fgmask_path)
                    mask_fg = torch.ceil(self.transform_mask(mask_fg)[0])

                feat_size = (self.imgsize_h // self.downsampling, self.imgsize_w // self.downsampling)
                mask_all = perlin_mask(image.shape, feat_size, 0, 6, mask_fg, 1)
                mask_s = torch.from_numpy(mask_all[0])
                mask_l = torch.from_numpy(mask_all[1])

                beta = np.random.normal(loc=self.mean, scale=self.std)
                beta = np.clip(beta, .2, .8)
                aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask_gt = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
        else:
            mask_gt = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "aug": aug_image,
            "mask_s": mask_s,
            "mask_gt": mask_gt,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        classpath = os.path.join(self.source, self.classname, self.split.value)
        maskpath = os.path.join(self.source, self.classname, "ground_truth")
        anomaly_types = os.listdir(classpath)

        imgpaths_per_class[self.classname] = {}
        maskpaths_per_class[self.classname] = {}

        for anomaly in anomaly_types:
            anomaly_path = os.path.join(classpath, anomaly)
            anomaly_files = sorted(os.listdir(anomaly_path))
            imgpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]

            if self.split == DatasetSplit.TEST and anomaly != "good":
                anomaly_mask_path = os.path.join(maskpath, anomaly)
                anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                maskpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]
            else:
                maskpaths_per_class[self.classname]["good"] = None

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
