"""
결함 데이터베이스 모듈

결함으로 판정된 탐지 결과를 SQLite DB에 저장합니다. (판정 기준: config.py ANOMALY_THRESHOLD)
DB 파일: defects.db (server/ 폴더에 자동 생성)
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defects.db")


def init_db():
    """DB 테이블 초기화 (없으면 생성)"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS defects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            source      TEXT    NOT NULL,  -- 'upload' or 'camera'
            model_type  TEXT    NOT NULL,  -- 'glass'
            filename    TEXT,              -- 업로드 파일명 (카메라는 NULL)
            anomaly_score   REAL  NOT NULL,
            inference_time  REAL,
            original_url    TEXT,
            overlay_url     TEXT
        )
    """)
    conn.commit()
    # 기존 DB에 inference_time 컬럼 없으면 추가
    cols = [r[1] for r in conn.execute("PRAGMA table_info(defects)").fetchall()]
    if "inference_time" not in cols:
        conn.execute("ALTER TABLE defects ADD COLUMN inference_time REAL")
        conn.commit()
    conn.close()


def save_defect(source: str, model_type: str, anomaly_score: float,
                inference_time: float = None,
                original_url: str = None, overlay_url: str = None,
                filename: str = None):
    """결함 레코드 저장. 호출 전 판정이 완료된 경우에만 호출할 것."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO defects (timestamp, source, model_type, filename, anomaly_score, inference_time, original_url, overlay_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source,
        model_type,
        filename,
        round(anomaly_score, 4),
        round(inference_time, 3) if inference_time is not None else None,
        original_url,
        overlay_url,
    ))
    conn.commit()
    conn.close()
    return True


def get_defects(limit: int = 100, min_score: float = 0.0,
                min_infer: float = None, max_infer: float = None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conditions = ["anomaly_score >= ?"]
    params = [min_score]
    if min_infer is not None:
        conditions.append("inference_time >= ?")
        params.append(min_infer)
    if max_infer is not None:
        conditions.append("inference_time <= ?")
        params.append(max_infer)
    params.append(limit)
    rows = conn.execute(f"""
        SELECT * FROM defects
        WHERE {' AND '.join(conditions)}
        ORDER BY id DESC
        LIMIT ?
    """, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_defect(defect_id: int):
    """ID로 결함 레코드 삭제. 삭제된 행 수 반환"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("DELETE FROM defects WHERE id = ?", (defect_id,))
    conn.commit()
    conn.close()
    return cur.rowcount


def get_defect_stats():
    """전체 통계 반환"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    stats = conn.execute("""
        SELECT
            COUNT(*)                        AS total_count,
            ROUND(AVG(anomaly_score), 4)    AS avg_score,
            ROUND(MAX(anomaly_score), 4)    AS max_score,
            ROUND(MIN(anomaly_score), 4)    AS min_score
        FROM defects
    """).fetchone()
    conn.close()
    return dict(stats) if stats else {}


# 서버 시작 시 자동 초기화
init_db()
