import os
import sqlite3
import threading
from datetime import datetime

HQ_DB_PATH = r"E:\SovereignEngine\AA_Dynamics_HQ\the_bridge\data_management\atlas_telemetry.db"
LEDGER_PATH = r"E:\SovereignEngine\AA_Dynamics_HQ\the_bridge\data_management\system_savings_ledger.json"


class HQTokenLogger:
    """Writes categorized gameplay LLM & Audio token metrics into AA Dynamics HQ atlas_telemetry.db."""

    def __init__(self, db_path: str = HQ_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.session_totals = {
            "total": 0,
            "Voice PTT": 0,
            "In-Game Hotkey": 0,
            "In-Game /atlas": 0,
            "Story Chronicle": 0,
            "Proactive / Hazard": 0,
            "Cockpit UI": 0,
            "Audio (STT/TTS)": 0,
        }
        self._ensure_tables()

    def _ensure_tables(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                c = conn.cursor()
                c.execute("PRAGMA journal_mode=WAL;")
                c.execute(
                    """CREATE TABLE IF NOT EXISTS token_logs
                       (timestamp TEXT, date TEXT, system TEXT, task_type TEXT, tokens INTEGER)"""
                )
                c.execute(
                    """CREATE TABLE IF NOT EXISTS gaming_token_telemetry (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       timestamp TEXT,
                       date TEXT,
                       trigger_source TEXT,
                       engine_mode TEXT,
                       specialist TEXT,
                       model_name TEXT,
                       prompt_tokens INTEGER,
                       completion_tokens INTEGER,
                       total_tokens INTEGER,
                       tts_chars INTEGER DEFAULT 0
                    )"""
                )
                conn.commit()
        except Exception as e:
            print(f"[HQTokenLogger] DB init warning: {e}")

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(round(len(text) / 3.8)))

    def log_llm_call(
        self,
        trigger_source: str,
        engine_mode: str,
        specialist: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict:
        total = max(1, int(prompt_tokens) + int(completion_tokens))
        now = datetime.now()
        ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")

        short_role = specialist.split("/")[0].strip()
        short_model = (model_name or "Local-GGUF").replace(".gguf", "")[:22]
        task_label = f"[{trigger_source} | {engine_mode}] {short_role} ({prompt_tokens} in / {completion_tokens} out) [{short_model}]"

        with self._lock:
            self.session_totals["total"] += total
            bucket = trigger_source if trigger_source in self.session_totals else "Cockpit UI"
            self.session_totals[bucket] = self.session_totals.get(bucket, 0) + total

            try:
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO token_logs (timestamp, date, system, task_type, tokens) VALUES (?, ?, ?, ?, ?)",
                        (ts_str, date_str, "Gaming Engine", task_label, total),
                    )
                    c.execute(
                        """INSERT INTO gaming_token_telemetry
                           (timestamp, date, trigger_source, engine_mode, specialist, model_name,
                            prompt_tokens, completion_tokens, total_tokens, tts_chars)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                        (
                            ts_str,
                            date_str,
                            trigger_source,
                            engine_mode,
                            specialist,
                            model_name,
                            int(prompt_tokens),
                            int(completion_tokens),
                            total,
                        ),
                    )
                    conn.commit()
            except Exception as e:
                print(f"[HQTokenLogger] Failed to write LLM telemetry: {e}")

        return dict(self.session_totals)

    def log_audio_event(self, audio_kind: str, role_or_source: str, text_content: str) -> dict:
        """Logs F5-TTS synthesis or Whisper STT transcription token-equivalent units into HQ DB."""
        clean = (text_content or "").strip()
        if not clean:
            return dict(self.session_totals)

        # Equivalent neural audio token units (~1 token per 4 chars synthesized/transcribed, min 15 units)
        equiv_units = max(15, int(round(len(clean) / 3.5)))
        now = datetime.now()
        ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        short_role = role_or_source.split("/")[0].strip()
        task_label = f"[{audio_kind}] {short_role} ({len(clean)} chars)"

        with self._lock:
            self.session_totals["total"] += equiv_units
            self.session_totals["Audio (STT/TTS)"] += equiv_units

            try:
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO token_logs (timestamp, date, system, task_type, tokens) VALUES (?, ?, ?, ?, ?)",
                        (ts_str, date_str, "Gaming Audio", task_label, equiv_units),
                    )
                    c.execute(
                        """INSERT INTO gaming_token_telemetry
                           (timestamp, date, trigger_source, engine_mode, specialist, model_name,
                            prompt_tokens, completion_tokens, total_tokens, tts_chars)
                           VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?)""",
                        (
                            ts_str,
                            date_str,
                            "Audio (STT/TTS)",
                            audio_kind,
                            role_or_source,
                            "F5-TTS / Whisper",
                            equiv_units,
                            len(clean),
                        ),
                    )
                    conn.commit()
            except Exception as e:
                print(f"[HQTokenLogger] Failed to write Audio telemetry: {e}")

        return dict(self.session_totals)