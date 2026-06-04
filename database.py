import sqlite3
import datetime

def init_db():
    conn = sqlite3.connect('samaya_logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS engagement_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  session_id TEXT,
                  emotion TEXT,
                  engagement_score REAL)''')
    conn.commit()
    conn.close()

def insert_log(session_id, emotion, score):
    conn = sqlite3.connect('samaya_logs.db')
    c = conn.cursor()
    c.execute("INSERT INTO engagement_logs (timestamp, session_id, emotion, engagement_score) VALUES (?, ?, ?, ?)",
              (datetime.datetime.now().isoformat(), session_id, emotion, score))
    conn.commit()
    conn.close()