"""
CogniOps AI - SQLite Database Repository for Decisions & Human Feedback.
"""
import sqlite3
import json
from typing import List, Dict, Any, Optional
from core.config import DB_PATH

def init_db(db_path: Optional[str] = None):
    """Initializes SQLite database with schema for decisions and feedback loop."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decision_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT UNIQUE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            situation_id TEXT,
            situation_title TEXT,
            recommended_strategy TEXT,
            recommended_strategy_title TEXT,
            expected_utility REAL,
            estimated_cost REAL,
            explanation TEXT,
            confidence_score REAL,
            user_status TEXT,  -- ACCEPTED, MODIFIED, REJECTED
            user_feedback TEXT,
            modified_strategy TEXT,
            realized_outcome REAL, -- Actual performance delta
            delta_utility REAL,    -- realized - expected
            agent_execution_time_sec REAL,
            raw_payload_json TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_decision_record(
    decision_id: str,
    situation_id: str,
    situation_title: str,
    recommended_strategy: str,
    recommended_strategy_title: str,
    expected_utility: float,
    estimated_cost: float,
    explanation: str,
    confidence_score: float,
    user_status: str,
    user_feedback: str,
    modified_strategy: Optional[str] = None,
    realized_outcome: Optional[float] = None,
    agent_execution_time_sec: float = 0.0,
    raw_payload: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None
) -> int:
    """Inserts or updates a decision record with human feedback and outcome."""
    init_db(db_path=db_path)
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    delta_utility = None
    if realized_outcome is not None:
        delta_utility = realized_outcome - expected_utility
        
    payload_str = json.dumps(raw_payload or {})
    
    cursor.execute("""
        INSERT OR REPLACE INTO decision_records (
            decision_id, situation_id, situation_title, recommended_strategy,
            recommended_strategy_title, expected_utility, estimated_cost,
            explanation, confidence_score, user_status, user_feedback,
            modified_strategy, realized_outcome, delta_utility,
            agent_execution_time_sec, raw_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        decision_id, situation_id, situation_title, recommended_strategy,
        recommended_strategy_title, expected_utility, estimated_cost,
        explanation, confidence_score, user_status, user_feedback,
        modified_strategy, realized_outcome, delta_utility,
        agent_execution_time_sec, payload_str
    ))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id

def get_all_decisions(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all decision records sorted chronologically."""
    init_db(db_path=db_path)
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM decision_records ORDER BY id DESC")
    rows = cursor.fetchall()
    results = [dict(r) for r in rows]
    conn.close()
    return results

def get_decision_by_id(decision_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves single decision record by unique ID."""
    init_db(db_path=db_path)
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM decision_records WHERE decision_id = ?", (decision_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
