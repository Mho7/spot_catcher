import os
import re
import time
import numpy as np
from pathlib import Path
from PIL import Image

# 리눅스 윈도우 호환 변환 코드
if os.name == 'nt':
    import anomalib.utils.path as _apath
    def _no_symlink(root_dir):
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        highest = -1
        for d in root_dir.iterdir():
            m = re.match(r'^v(\d+)$', d.name)
            if m:
                highest = max(highest, int(m.group(1)))
        new_dir = root_dir / f"v{highest + 1}"
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir
    _apath.create_versioned_dir = _no_symlink

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


#  합성 데이터 생성-> autoencoder 와 동일한 이미지를 가져다씀
DATA_ROOT = Path("study/ai/synth_data")

def make_synth_data(root: Path, n_train=30, n_test_good=10, n_test_bad=10):
    if (root / "train" / "good" / "0.png").exists():
        return  # 이미 생성된 경우 넘어가버리기

    rng = np.random.default_rng(42)
    size = (128, 128)

    dirs = {
        "train/good":  (n_train,     False),
        "test/good":   (n_test_good, False),
        "test/bad":    (n_test_bad,  True),
    }

    for rel, (n, is_defect) in dirs.items():
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            base = rng.integers(100, 160, size + (3,), dtype=np.uint8)
            noise = rng.integers(0, 20, size + (3,), dtype=np.uint8)
            img = np.clip(base.astype(int) + noise - 10, 0, 255).astype(np.uint8)

            if is_defect:
                cx, cy = rng.integers(20, 108, 2)
                r = rng.integers(8, 20)
                Y, X = np.ogrid[:128, :128]
                mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
                img[mask] = np.clip(img[mask].astype(int) + 100, 0, 255).astype(np.uint8)

            Image.fromarray(img).save(d / f"{i}.png")

    print(f" 이미지 무작위로다가 생성 완료: {root}")

make_synth_data(DATA_ROOT)

datamodule = Folder(
    name="synth",
    root=DATA_ROOT,
    normal_dir="train/good",
    abnormal_dir="test/bad",
    normal_test_dir="test/good",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(max_epochs=1)

print("PatchCore 학습 시작")
train_start = time.time()
engine.fit(model=model, datamodule=datamodule)
train_time = time.time() - train_start
print(f"학습 시간: {train_time:.1f}초")

print("정확도 평가")
test_results = engine.test(model=model, datamodule=datamodule)
if test_results:
    for key, val in test_results[0].items():
        print(f"  {key}: {val:.4f}")

print("추론 속도 측정")
infer_start = time.time()
predictions = engine.predict(model=model, datamodule=datamodule)
infer_time = time.time() - infer_start
n_images = len(predictions) if predictions else 1
print(f"  총 추론 시간: {infer_time:.3f}초")
print(f"  이미지 수: {n_images}")
print(f"  이미지당 평균: {infer_time / n_images * 1000:.1f}ms")