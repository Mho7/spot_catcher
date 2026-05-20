"""
재학습 파이프라인 전용 DB 함수 (sik_develop에서만 사용)

desktop 브랜치의 database.py를 오염시키지 않기 위해 별도 모듈로 분리.
같은 defects.db 파일을 공유한다.
"""
import contextlib
import sqlite3
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from database import DB_PATH


def init_training_tables():
    """재학습용 테이블 초기화 (없으면 생성)."""
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_images (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath    TEXT    NOT NULL UNIQUE,
                label       TEXT    NOT NULL,   -- 'normal' or 'defect'
                discovered  TEXT    NOT NULL,
                run_id      INTEGER             -- 사용된 run id (NULL = 미사용)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at      TEXT    NOT NULL,
                finished_at     TEXT,
                status          TEXT    NOT NULL,  -- 'running'|'completed'|'failed'
                normal_count    INTEGER,
                defect_count    INTEGER,
                image_auroc     REAL,
                checkpoint_path TEXT,
                notes           TEXT
            )
        """)
        conn.commit()


def register_images(filepaths_labels: list):
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.executemany(
            "INSERT OR IGNORE INTO training_images (filepath, label, discovered) VALUES (?, ?, ?)",
            [(fp, label, now) for fp, label in filepaths_labels],
        )
        conn.commit()


def get_unused_normal_count() -> int:
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM training_images WHERE label='normal' AND run_id IS NULL"
        ).fetchone()
    return row[0] if row else 0


def get_unused_images() -> dict:
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT filepath, label FROM training_images WHERE run_id IS NULL"
        ).fetchall()
    result = {"normal": [], "defect": []}
    for r in rows:
        result[r["label"]].append(r["filepath"])
    return result


def create_training_run(normal_count: int, defect_count: int) -> int:
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(
            "INSERT INTO training_runs (started_at, status, normal_count, defect_count) VALUES (?, 'running', ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), normal_count, defect_count),
        )
        conn.commit()
        return cur.lastrowid


def finish_training_run(run_id, status, image_auroc=None, checkpoint_path=None, notes=None):
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "UPDATE training_runs SET finished_at=?, status=?, image_auroc=?, checkpoint_path=?, notes=? WHERE id=?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, image_auroc, checkpoint_path, notes, run_id),
        )
        conn.commit()


def mark_images_used(run_id: int):
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("UPDATE training_images SET run_id=? WHERE run_id IS NULL", (run_id,))
        conn.commit()


def get_training_history(limit: int = 20) -> list:
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# 모듈 로드 시 테이블 자동 생성
init_training_tables()
