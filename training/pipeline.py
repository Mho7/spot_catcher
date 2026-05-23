"""재학습 파이프라인 (단독 실행 또는 watcher에서 호출)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, os.path.dirname(__file__))

from config import GLASS_INPUT_SIZE
from db_training import (
    get_unused_normal_count, create_training_run,
    finish_training_run, mark_images_used,
)
from data_manager import scan_and_register, prepare_split
from dataset import NormalImageDataset, EvalDataset
from trainer import run_finetune

TRIGGER_THRESHOLD = 100


def run(meta_epochs=100, batch_size=4, force=False):
    print("[파이프라인] 시작")

    n_normal, n_defect = scan_and_register()
    print(f"[파이프라인] 스캔: 정상={n_normal}, 결함={n_defect}")

    unused = get_unused_normal_count()
    print(f"[파이프라인] 미사용 정상 이미지: {unused}장 / 임계값: {TRIGGER_THRESHOLD}장")

    if unused < TRIGGER_THRESHOLD and not force:
        print("[파이프라인] 조건 미충족 — 종료")
        return {"triggered": False}

    split = prepare_split()
    run_id = create_training_run(
        normal_count=len(split["train_normal"]) + len(split["val_normal"]),
        defect_count=len(split["val_defect"]),
    )
    print(f"[파이프라인] run_id={run_id}, 학습={len(split['train_normal'])}, "
          f"val 정상={len(split['val_normal'])}, val 결함={len(split['val_defect'])}")

    train_ds = NormalImageDataset(
        split["train_normal"], input_size=GLASS_INPUT_SIZE,
        defect_paths=split.get("train_defect"),
        defect_mask_paths=split.get("train_defect_mask"),
    )
    eval_ds  = EvalDataset(split["val_normal"], split["val_defect"], input_size=GLASS_INPUT_SIZE)

    result = run_finetune(train_ds, eval_ds, run_id=run_id,
                          meta_epochs=meta_epochs, batch_size=batch_size)

    finish_training_run(
        run_id=run_id,
        status=result["status"],
        image_auroc=result.get("image_auroc"),
        checkpoint_path=result.get("checkpoint_path"),
        notes=result.get("notes"),
    )
    mark_images_used(run_id)

    print(f"[파이프라인] {result['status']} — {result.get('notes')}")
    if result["status"] == "completed":
        auroc = result.get("image_auroc")
        print(f"[파이프라인] image_auroc={auroc:.4f}" if auroc else "[파이프라인] auroc N/A")
        print(f"[파이프라인] 새 체크포인트: {result['checkpoint_path']}")
        print("[파이프라인] 서버 재시작 시 자동 적용됩니다.")

    return {"triggered": True, **result}
