"""GLASS 파인튜닝: 기존 체크포인트에서 새 데이터로 추가 학습."""
import os
import sys
import glob
import re
import shutil

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from config import (
    SAVE_DIR, GLASS_BACKBONE, GLASS_LAYERS, GLASS_INPUT_SIZE,
    GLASS_PRETRAIN_EMBED_DIM, GLASS_TARGET_EMBED_DIM,
    GLASS_PATCHSIZE, GLASS_DSC_LAYERS, GLASS_DSC_HIDDEN, GLASS_PRE_PROJ,
)
from models.glass_detector import find_glass_checkpoint
from vendor.glass import GLASS, backbones


def _current_max_epoch():
    candidates = glob.glob(os.path.join(SAVE_DIR, "ckpt_best_*.pth"))
    if not candidates:
        return 0
    def _ep(p):
        m = re.search(r"ckpt_best_(\d+)\.pth", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return max(_ep(p) for p in candidates)


def _build_model(device):
    h, w = GLASS_INPUT_SIZE
    backbone = backbones.load(GLASS_BACKBONE)
    model = GLASS(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=GLASS_LAYERS,
        device=device,
        input_shape=(3, h, w),
        pretrain_embed_dimension=GLASS_PRETRAIN_EMBED_DIM,
        target_embed_dimension=GLASS_TARGET_EMBED_DIM,
        patchsize=GLASS_PATCHSIZE,
        dsc_layers=GLASS_DSC_LAYERS,
        dsc_hidden=GLASS_DSC_HIDDEN,
        pre_proj=GLASS_PRE_PROJ,
        meta_epochs=0,
    )
    return model


def run_finetune(train_dataset, eval_dataset, run_id,
                 meta_epochs=100, batch_size=4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        current_ckpt = find_glass_checkpoint()
    except FileNotFoundError as e:
        return {"status": "failed", "checkpoint_path": None, "image_auroc": None, "notes": str(e)}

    model = _build_model(device)
    state = torch.load(current_ckpt, map_location=device, weights_only=False)
    if "discriminator" in state:
        model.discriminator.load_state_dict(state["discriminator"])
        if "pre_projection" in state and GLASS_PRE_PROJ > 0:
            model.pre_projection.load_state_dict(state["pre_projection"])
    else:
        model.load_state_dict(state, strict=False)

    run_name = f"finetune_{run_id}"
    ckpt_dir = os.path.join(SAVE_DIR, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(current_ckpt, os.path.join(ckpt_dir, "ckpt.pth"))

    model.set_model_dir(SAVE_DIR, run_name)
    model.meta_epochs = meta_epochs

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    try:
        result = model.trainer(train_loader, eval_loader, run_name)
    except Exception as e:
        return {"status": "failed", "checkpoint_path": None, "image_auroc": None, "notes": f"trainer() 오류: {e}"}

    best_files = glob.glob(os.path.join(ckpt_dir, "ckpt_best_*.pth"))
    if not best_files:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        return {"status": "failed", "checkpoint_path": None, "image_auroc": None, "notes": "성능 개선 없음 — best 체크포인트 미생성"}

    image_auroc = float(result[0]) if isinstance(result, (list, tuple)) and len(result) >= 1 else None

    new_epoch = _current_max_epoch() + 1
    new_ckpt_path = os.path.join(SAVE_DIR, f"ckpt_best_{new_epoch}.pth")
    shutil.copy(best_files[0], new_ckpt_path)
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    return {
        "status": "completed",
        "checkpoint_path": new_ckpt_path,
        "image_auroc": image_auroc,
        "notes": f"epoch {new_epoch} 저장 완료",
    }
