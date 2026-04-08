"""
결함 데이터베이스 모듈

결함률 30% 이상(anomaly_score >= 0.3)인 탐지 결과를 SQLite DB에 저장합니다.
DB 파일: defects.db (프로젝트 루트에 자동 생성)
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defects.db")

# 30% 이상일 때 저장
DEFECT_DB_THRESHOLD = 0.3


def init_db():
    """DB 테이블 초기화 (없으면 생성)"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS defects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            source      TEXT    NOT NULL,  -- 'upload' or 'camera'
            model_type  TEXT    NOT NULL,  -- 'patchcore' or 'autoencoder'
            filename    TEXT,              -- 업로드 파일명 (카메라는 NULL)
            anomaly_score REAL  NOT NULL,
            original_url  TEXT,
            overlay_url   TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_defect(source: str, model_type: str, anomaly_score: float,
                original_url: str = None, overlay_url: str = None,
                filename: str = None):
    """
    결함률 30% 이상인 경우 DB에 저장

    Returns:
        True  — 저장됨
        False — 임계값 미달로 저장 안 함
    """
    if anomaly_score < DEFECT_DB_THRESHOLD:
        return False

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO defects (timestamp, source, model_type, filename, anomaly_score, original_url, overlay_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source,
        model_type,
        filename,
        round(anomaly_score, 4),
        original_url,
        overlay_url,
    ))
    conn.commit()
    conn.close()
    return True


def get_defects(limit: int = 100, min_score: float = DEFECT_DB_THRESHOLD):
    """
    저장된 결함 데이터 조회

    Args:
        limit: 최대 조회 개수 (최신순)
        min_score: 최소 anomaly_score 필터
    Returns:
        list of dict
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM defects
        WHERE anomaly_score >= ?
        ORDER BY id DESC
        LIMIT ?
    """, (min_score, limit)).fetchall()
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
