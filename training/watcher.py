"""
재학습 자동 트리거 (sik_develop에서 백그라운드로 실행)

실행:
    conda run -n spot python training/watcher.py

매 CHECK_INTERVAL 초마다 두 조건을 확인하고 모두 충족하면 파이프라인 실행:
    1. 미사용 정상 이미지 >= TRIGGER_THRESHOLD
    2. 마지막 학습 완료 후 MIN_RETRAIN_HOURS 이상 경과
"""
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, os.path.dirname(__file__))

from db_training import get_unused_normal_count, get_training_history
import pipeline as pl

# ── 설정 ──────────────────────────────────────────────────────────────
CHECK_INTERVAL     = 3600   # 조건 확인 주기 (초) — 기본 1시간
TRIGGER_THRESHOLD  = 100    # 재학습 시작 최소 정상 이미지 수
MIN_RETRAIN_HOURS  = 12     # 마지막 학습 후 최소 대기 시간 (시간)
META_EPOCHS        = 100    # 파인튜닝 epoch 수
BATCH_SIZE         = 4
# ──────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watcher] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "watcher.log")),
    ],
)
log = logging.getLogger(__name__)


def _last_completed_run_time() -> datetime | None:
    """가장 최근 completed 학습의 완료 시각 반환. 없으면 None."""
    history = get_training_history(limit=10)
    for run in history:
        if run["status"] == "completed" and run["finished_at"]:
            return datetime.strptime(run["finished_at"], "%Y-%m-%d %H:%M:%S")
    return None


def _should_retrain() -> tuple[bool, str]:
    """재학습 조건 확인. (실행 여부, 이유) 반환."""
    unused = get_unused_normal_count()
    if unused < TRIGGER_THRESHOLD:
        return False, f"이미지 부족 ({unused}/{TRIGGER_THRESHOLD})"

    last_run = _last_completed_run_time()
    if last_run is not None:
        elapsed = datetime.now() - last_run
        if elapsed < timedelta(hours=MIN_RETRAIN_HOURS):
            remaining = timedelta(hours=MIN_RETRAIN_HOURS) - elapsed
            return False, f"최소 간격 미충족 (남은 시간: {str(remaining).split('.')[0]})"

    return True, f"조건 충족 (미사용 이미지: {unused}장)"


def run_loop(check_interval: int = CHECK_INTERVAL):
    log.info(f"watcher 시작 — 확인 주기: {check_interval}s, "
             f"임계값: {TRIGGER_THRESHOLD}장, 최소 간격: {MIN_RETRAIN_HOURS}h")

    while True:
        should, reason = _should_retrain()
        log.info(f"조건 확인: {reason}")

        if should:
            log.info("재학습 파이프라인 시작")
            try:
                result = pl.run(meta_epochs=META_EPOCHS, batch_size=BATCH_SIZE)
                if result.get("status") == "completed":
                    log.info(f"재학습 완료 — {result.get('notes')} "
                             f"(auroc={result.get('image_auroc')})")
                    log.info("서버 재시작 시 새 모델이 자동 적용됩니다.")
                elif result.get("triggered") is False:
                    log.info("파이프라인 조건 미충족 (이미지 부족)")
                else:
                    log.warning(f"재학습 실패: {result.get('notes')}")
            except Exception as e:
                log.error(f"파이프라인 예외: {e}", exc_info=True)

        log.info(f"다음 확인까지 {check_interval}초 대기")
        time.sleep(check_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="재학습 자동 트리거")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL,
                        help="조건 확인 주기 (초, 기본 3600)")
    args = parser.parse_args()
    run_loop(check_interval=args.interval)
