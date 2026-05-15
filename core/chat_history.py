import sqlite3
import os
from configs.settings import get_db_path

def init_db():
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS global_chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_global_message(role, content):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute(
        "INSERT INTO global_chat (role, content) VALUES (?, ?)",
        (role, content)
    )
    conn.commit()
    conn.close()

def get_global_history(limit=20):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute('''
        SELECT role, content
        FROM global_chat
        ORDER BY id ASC
        LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]

init_db()