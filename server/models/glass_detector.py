import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import (
    GLASS_BACKBONE,
    GLASS_DSC_HIDDEN,
    GLASS_DSC_LAYERS,
    GLASS_INPUT_SIZE,
    GLASS_LAYERS,
    GLASS_PATCHSIZE,
    GLASS_PRE_PROJ,
    GLASS_PRETRAIN_EMBED_DIM,
    GLASS_TARGET_EMBED_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SAVE_DIR,
)
from vendor.glass import GLASS, backbones


class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)


def find_glass_checkpoint(path=None, save_dir=SAVE_DIR):
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"GLASS 체크포인트가 없습니다: {path}")
        return path

    candidates = sorted(glob.glob(os.path.join(save_dir, "ckpt_best_*.pth")))
    if not candidates:
        raise FileNotFoundError(f"GLASS 체크포인트가 없습니다: {save_dir}/ckpt_best_*.pth")
    return candidates[0]


class GlassDetector:
    def __init__(self, checkpoint_path=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.checkpoint_path = find_glass_checkpoint(checkpoint_path)
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _build_model(self):
        height, width = GLASS_INPUT_SIZE

        backbone = backbones.load(GLASS_BACKBONE)
        model = GLASS(self.device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=GLASS_LAYERS,
            device=self.device,
            input_shape=(3, height, width),
            pretrain_embed_dimension=GLASS_PRETRAIN_EMBED_DIM,
            target_embed_dimension=GLASS_TARGET_EMBED_DIM,
            patchsize=GLASS_PATCHSIZE,
            dsc_layers=GLASS_DSC_LAYERS,
            dsc_hidden=GLASS_DSC_HIDDEN,
            pre_proj=GLASS_PRE_PROJ,
        )

        state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if "discriminator" in state_dict:
            model.discriminator.load_state_dict(state_dict["discriminator"])
            if "pre_projection" in state_dict and GLASS_PRE_PROJ > 0:
                model.pre_projection.load_state_dict(state_dict["pre_projection"])
        else:
            model.load_state_dict(state_dict, strict=False)

        return model

    @staticmethod
    def _build_transform():
        height, width = GLASS_INPUT_SIZE
        return transforms.Compose([
            transforms.Resize((height, width)),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def predict(self, image):
        height, width = GLASS_INPUT_SIZE
        pil_image = image.convert("RGB")
        original_np = np.array(pil_image.resize((width, height)))
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        scores, masks = self.model._predict(tensor)
        return float(scores[0]), np.array(masks[0]), original_np
