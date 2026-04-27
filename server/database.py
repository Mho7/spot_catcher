"""
결함 데이터베이스 모듈

결함률 30% 이상(anomaly_score >= 0.3)인 탐지 결과를 SQLite DB에 저장합니다.
DB 파일: defects.db (server/ 폴더에 자동 생성)
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defects.db")

# 30% 이상일 때 저장
DEFECT_DB_THRESHOLD = 0.3


def init_db():
    """DB 테이블 초기화 (없으면 생성). 기존 DB에 inference_time 컬럼이 없으면 추가."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS defects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            source      TEXT    NOT NULL,  -- 'upload' or 'camera'
            model_type  TEXT    NOT NULL,  -- 'patchcore'
            filename    TEXT,              -- 업로드 파일명 (카메라는 NULL)
            anomaly_score REAL  NOT NULL,
            inference_time REAL,           -- 추론 소요 시간(초)
            original_url  TEXT,
            overlay_url   TEXT
        )
    """)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(defects)").fetchall()]
    if "inference_time" not in cols:
        conn.execute("ALTER TABLE defects ADD COLUMN inference_time REAL")
    conn.commit()
    conn.close()


def save_defect(source: str, model_type: str, anomaly_score: float,
                original_url: str = None, overlay_url: str = None,
                filename: str = None, inference_time: float = None,
                force: bool = False):
    """
    결함률 30% 이상이면 자동 저장. force=True면 임계값 무시하고 강제 저장.

    Returns:
        True  — 저장됨
        False — 임계값 미달로 저장 안 함 (force=False 한정)
    """
    if not force and anomaly_score < DEFECT_DB_THRESHOLD:
        return False

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


def get_defects(
    limit: int = 100,
    min_score: float = DEFECT_DB_THRESHOLD,
    max_score: float = None,
    start_date: str = None,
    end_date: str = None,
    min_inference_time: float = None,
    max_inference_time: float = None,
):
    """
    저장된 결함 데이터 조회 (검색/필터 지원)

    Args:
        limit              : 최대 조회 개수 (최신순)
        min_score          : 최소 anomaly_score (기본 0.3)
        max_score          : 최대 anomaly_score
        start_date         : 시작일 "YYYY-MM-DD" (해당일 00:00:00 이상)
        end_date           : 종료일 "YYYY-MM-DD" (해당일 23:59:59 이하)
        min_inference_time : 최소 추론 시간(초)
        max_inference_time : 최대 추론 시간(초)
    Returns:
        list of dict
    """
    where = ["anomaly_score >= ?"]
    params = [min_score]

    if max_score is not None:
        where.append("anomaly_score <= ?")
        params.append(max_score)
    if start_date:
        where.append("timestamp >= ?")
        params.append(start_date if " " in start_date else start_date + " 00:00:00")
    if end_date:
        where.append("timestamp <= ?")
        params.append(end_date if " " in end_date else end_date + " 23:59:59")
    if min_inference_time is not None:
        where.append("inference_time >= ?")
        params.append(min_inference_time)
    if max_inference_time is not None:
        where.append("inference_time <= ?")
        params.append(max_inference_time)

    sql = f"""
        SELECT * FROM defects
        WHERE {' AND '.join(where)}
        ORDER BY id DESC
        LIMIT ?
    """
    params.append(limit)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
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
        WHERE anomaly_score >= ?
    """, (DEFECT_DB_THRESHOLD,)).fetchone()
    conn.close()
    return dict(stats) if stats else {}


# 서버 시작 시 자동 초기화
init_db()
