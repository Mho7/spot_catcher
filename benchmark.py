"""
PatchCore 경량 모델 추론 속도 벤치마크.

CPU 환경에서 단일 이미지 추론을 N회 반복 측정하고,
HCI 학술 기준에 따라 양호 / 주의 / 나쁨 3단계로 자동 분류한다.

사용 예:
    python benchmark.py --image server/static/cam_2258b188.png
    python benchmark.py --image-dir server/static/ --runs 200 --warmup 10

요구 라이브러리: torch, torchvision, numpy, pillow
"""
import os
import sys
import time
import argparse
import pickle
import platform
import datetime
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image

# CPU 환경 명시
torch.set_num_threads(os.cpu_count())

# config/benchmark_criteria 임포트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.benchmark_criteria import (
    THRESHOLD_GOOD_MS,
    THRESHOLD_CAUTION_MS,
    classify_latency,
)
from config import benchmark_criteria as _criteria_mod

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "server", "saved_models", "patchcore.pkl"
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
STAGE_NAMES = ["load_preprocess", "feature_extract", "nn_search", "score"]


def load_model(pkl_path):
    """pkl 메타데이터에 맞춰 backbone + memory_bank 준비."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    backbone_name = data.get("backbone", "resnet18")
    layers = data.get("layers", ["layer2"])
    image_size = tuple(data.get("image_size", (144, 256)))
    memory_bank = torch.from_numpy(data["memory_bank"]).float()

    full_model = getattr(models, backbone_name)(weights="IMAGENET1K_V1")
    return_nodes = {layer: layer for layer in layers}
    backbone = create_feature_extractor(full_model, return_nodes=return_nodes)
    backbone.eval()

    mb_sq = (memory_bank ** 2).sum(dim=1, keepdim=True).T  # [1, M], 사전 계산

    return {
        "backbone": backbone,
        "backbone_name": backbone_name,
        "layers": layers,
        "image_size": image_size,
        "memory_bank": memory_bank,
        "memory_bank_sq": mb_sq,
    }


def make_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def run_one(image_path, model, transform):
    """이미지 한 장 추론 후 단계별 latency(ms) dict + score 반환."""
    timings = {}

    # 1) 이미지 로드 + 전처리
    t0 = time.perf_counter()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    timings["load_preprocess"] = (time.perf_counter() - t0) * 1000

    # 2) Feature 추출 (ResNet18 layer2 등)
    t0 = time.perf_counter()
    with torch.no_grad():
        feats = model["backbone"](tensor)
    feat_list, target_size = [], None
    for layer_name in model["layers"]:
        f = feats[layer_name]
        if target_size is None:
            target_size = f.shape[2:]
        if f.shape[2:] != target_size:
            f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
        feat_list.append(f)
    combined = torch.cat(feat_list, dim=1)
    combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)
    B, C, H, W = combined.shape
    patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
    patches = F.normalize(patches, dim=-1)
    patches_flat = patches.reshape(-1, patches.shape[-1])
    timings["feature_extract"] = (time.perf_counter() - t0) * 1000

    # 3) Memory bank nearest neighbor 검색
    t0 = time.perf_counter()
    mb = model["memory_bank"]
    mb_sq = model["memory_bank_sq"]
    chunk_size = 512
    chunks = []
    with torch.no_grad():
        for i in range(0, patches_flat.shape[0], chunk_size):
            chunk = patches_flat[i:i + chunk_size]
            chunk_sq = (chunk ** 2).sum(dim=1, keepdim=True)
            cross = chunk @ mb.T
            d_sq = torch.clamp(chunk_sq + mb_sq - 2 * cross, min=0)
            chunks.append(d_sq.sqrt().min(dim=1).values)
    distances = torch.cat(chunks).numpy()
    timings["nn_search"] = (time.perf_counter() - t0) * 1000

    # 4) Anomaly score
    t0 = time.perf_counter()
    top_k = min(5, len(distances))
    score = float(np.sort(distances)[-top_k:].mean())
    timings["score"] = (time.perf_counter() - t0) * 1000

    timings["total"] = sum(timings[k] for k in STAGE_NAMES)
    return timings, score


def stats(values):
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "stdev":  float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
    }


def fmt_stats(s):
    return (f"mean {s['mean']:8.2f} | median {s['median']:8.2f} | "
            f"stdev {s['stdev']:7.2f} | p95 {s['p95']:8.2f} | p99 {s['p99']:8.2f}")


def get_ram_str():
    """psutil 없이 시스템 RAM 추정."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return f"{int(line.split()[1]) / (1024**2):.1f} GB"
    except Exception:
        pass
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2).decode().strip()
        return f"{int(out) / (1024**3):.1f} GB"
    except Exception:
        pass
    return "N/A"


def get_cpu_str():
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=2
        ).decode().strip()
        if out:
            return out
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "Unknown"


def get_env_info():
    return {
        "cpu": get_cpu_str(),
        "cpu_cores": os.cpu_count(),
        "ram": get_ram_str(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
    }


def write_report(report_path, env, conditions, stage_stats, total_stats, grade, improvement_ms):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# PatchCore 경량 모델 벤치마크 보고서")
    lines.append("")
    lines.append(f"생성일시: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 측정 환경")
    lines.append(f"- CPU: {env['cpu']} ({env['cpu_cores']} threads)")
    lines.append(f"- RAM: {env['ram']}")
    lines.append(f"- OS: {env['os']}")
    lines.append(f"- Python: {env['python']}")
    lines.append(f"- PyTorch: {env['pytorch']}")
    lines.append("")
    lines.append("## 측정 조건")
    lines.append(f"- 이미지 수: {conditions['n_images']}")
    lines.append(f"- runs: {conditions['runs']}")
    lines.append(f"- warmup: {conditions['warmup']}")
    lines.append(f"- 모델 경로: `{conditions['model_path']}`")
    lines.append(f"- backbone: {conditions['backbone']}")
    lines.append(f"- 입력 크기: {conditions['image_size']}")
    lines.append("")
    lines.append("## 결과")
    lines.append("")
    lines.append("### 단계별 latency (ms)")
    lines.append("| 단계 | mean | median | stdev | p95 | p99 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in STAGE_NAMES:
        s = stage_stats.get(name, {})
        if s:
            lines.append(f"| {name} | {s['mean']:.2f} | {s['median']:.2f} | "
                         f"{s['stdev']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} |")
    lines.append("")
    lines.append("### 종합 (total) latency (ms)")
    lines.append(f"- mean:   {total_stats['mean']:.2f}")
    lines.append(f"- median: {total_stats['median']:.2f}")
    lines.append(f"- stdev:  {total_stats['stdev']:.2f}")
    lines.append(f"- p95:    {total_stats['p95']:.2f}")
    lines.append(f"- p99:    {total_stats['p99']:.2f}")
    lines.append("")
    lines.append(f"### 등급 판정: **{grade}**")
    if improvement_ms > 0:
        lines.append(f"- 양호 등급까지 **{improvement_ms:.0f} ms** 단축 필요")
    else:
        lines.append("- 이미 양호 등급 (추가 단축 불필요)")
    lines.append("")
    lines.append("## 등급 기준")
    lines.append(f"| 등급 | 조건 |")
    lines.append(f"|---|---|")
    lines.append(f"| 양호 | t ≤ {THRESHOLD_GOOD_MS} ms |")
    lines.append(f"| 주의 | {THRESHOLD_GOOD_MS} < t ≤ {THRESHOLD_CAUTION_MS} ms |")
    lines.append(f"| 나쁨 | t > {THRESHOLD_CAUTION_MS} ms |")
    lines.append("")
    lines.append("## 학술 출처 (APA 형식)")
    lines.append("")
    lines.append((_criteria_mod.__doc__ or "").strip())

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def collect_images(args):
    images = []
    if args.image:
        images.append(args.image)
    if args.image_dir:
        d = Path(args.image_dir)
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                images.append(str(p))
    return images


def main():
    parser = argparse.ArgumentParser(description="PatchCore 경량 모델 추론 속도 벤치마크")
    parser.add_argument("--image", type=str, help="단일 이미지 경로")
    parser.add_argument("--image-dir", type=str, help="이미지 디렉토리 (jpg/png/bmp)")
    parser.add_argument("--runs", type=int, default=100, help="측정 반복 횟수 (기본 100)")
    parser.add_argument("--warmup", type=int, default=5, help="워밍업 횟수 (기본 5)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="pkl 모델 경로")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("--image 또는 --image-dir 중 하나는 필수입니다.")

    images = collect_images(args)
    if not images:
        parser.error("이미지를 찾지 못했습니다.")

    print("=" * 60)
    print("PatchCore 경량 모델 추론 속도 벤치마크")
    print("=" * 60)
    print(f"모델 : {args.model}")
    print(f"이미지: {len(images)}개 | runs: {args.runs} | warmup: {args.warmup}")
    print(f"CPU threads (torch): {torch.get_num_threads()}")
    print()

    model = load_model(args.model)
    print(f"backbone   : {model['backbone_name']}")
    print(f"layers     : {model['layers']}")
    print(f"image_size : {model['image_size']}")
    print(f"memory_bank: {tuple(model['memory_bank'].shape)}")
    print()

    transform = make_transform(model["image_size"])

    print(f"warmup ({args.warmup} runs)...")
    for i in range(args.warmup):
        run_one(images[i % len(images)], model, transform)

    print(f"measuring ({args.runs} runs)...")
    buckets = {k: [] for k in STAGE_NAMES + ["total"]}
    for i in range(args.runs):
        t, _ = run_one(images[i % len(images)], model, transform)
        for k in buckets:
            buckets[k].append(t[k])

    stage_stats = {k: stats(buckets[k]) for k in STAGE_NAMES}
    total_stats = stats(buckets["total"])

    print()
    print("─" * 60)
    print("단계별 latency (ms)")
    print("─" * 60)
    for name in STAGE_NAMES:
        print(f"  {name:18s}: {fmt_stats(stage_stats[name])}")
    print()
    print("─" * 60)
    print("종합 latency (ms)")
    print("─" * 60)
    print(f"  {'total':18s}: {fmt_stats(total_stats)}")
    print()

    grade = classify_latency(total_stats["mean"])
    improvement_ms = max(0.0, total_stats["mean"] - THRESHOLD_GOOD_MS)

    print(f"=== 등급 판정: {grade} ===")
    if improvement_ms > 0:
        print(f"양호 등급까지 {improvement_ms:.0f} ms 단축 필요")
    else:
        print("이미 양호 등급 (추가 단축 불필요)")
    print()

    env = get_env_info()
    conditions = {
        "n_images": len(images),
        "runs": args.runs,
        "warmup": args.warmup,
        "model_path": args.model,
        "backbone": model["backbone_name"],
        "image_size": str(model["image_size"]),
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "reports", f"benchmark_{ts}.md"
    )
    written = write_report(report_path, env, conditions, stage_stats,
                           total_stats, grade, improvement_ms)
    print(f"보고서 저장: {written}")


if __name__ == "__main__":
    main()
