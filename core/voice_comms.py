import os
import re
import sys
import time
import io
import math
import json
import wave
import queue
import ctypes
import tempfile
import threading
import subprocess
import winsound
from collections import deque
import requests
import soundfile as sf

ENGINE_ROOT = r"E:\SovereignEngine\AA_Dynamics_Gaming_Engine"
VOICES_DIR = os.path.join(ENGINE_ROOT, "voices")
PROFILES_JSON = os.path.join(VOICES_DIR, "voice_profiles.json")
F5_TTS_URL = "http://127.0.0.1:1237/v1/audio/generate"
F5_HEALTH_URL = "http://127.0.0.1:1237/health"

PTT_KEYS = {
    "~ (Grave) / Mouse4": (0xC0, 0x05),
    "~ (Grave Only)":     (0xC0,),
    "Mouse4 (Side Btn)":  (0x05,),
    "V Key":              (0x56,),
    "Caps Lock":          (0x14,),
    "F4 Key":             (0x73,),
}

TASK_HUD_KEYS = {
    "J Key (Task Menu)":  0x4A,
    "F6 Key (Task Menu)": 0x75,
    "K Key (Task Menu)":  0x4B,
}

DEFAULT_VOICE_SPECS = {
    "Geologist / Navigator": {
        "stem": "geologist",
        "ref_text": "Radar lock confirmed. Target ore vein center is forty blocks southwest at depth level sixty.",
        "speed": 1.04,
    },
    "Wayfinder / Map Specialist": {
        "stem": "wayfinder",
        "ref_text": "Waypoint locked on your map. Bear north-northwest across the ridge—we're entering a high-humidity biome two hundred blocks out.",
        "speed": 1.04,
    },
    "Chief Industrial Planner": {
        "stem": "planner",
        "ref_text": "Production schedule updated. Electric blast furnace and high voltage chemical reactor are operating nominal.",
        "speed": 1.02,
    },
    "Weapon / Armor / Tool Specialist": {
        "stem": "armorer",
        "ref_text": "Hold up on that alloy. Your blast furnace isn't rated for steel yet, so swap in a wrought iron tool rod and level the pickaxe first.",
        "speed": 1.02,
    },
    "Combat Specialist": {
        "stem": "combat",
        "ref_text": "Watch your spacing! You're dropping speed in tight terrain. Kite back, get blocks between you and that infernal mob, and heal up.",
        "speed": 1.06,
    },
    "Safety & Voltage Officer": {
        "stem": "safety",
        "ref_text": "Warning! High voltage amperage hazard detected on the main power line. Isolate the machine hull immediately.",
        "speed": 1.05,
    },
    "Story Chronicler": {
        "stem": "chronicler",
        "ref_text": "The wind howled across the dark frontier as the lone builder laid the first stone of the new settlement.",
        "speed": 0.98,
    },
}


def ensure_voice_profiles() -> dict:
    os.makedirs(VOICES_DIR, exist_ok=True)
    profiles = {}
    if os.path.isfile(PROFILES_JSON):
        try:
            with open(PROFILES_JSON, "r", encoding="utf-8") as f:
                profiles = json.load(f)
        except Exception:
            profiles = {}

    changed = False
    for role, spec in DEFAULT_VOICE_SPECS.items():
        stem = spec["stem"]
        wav_path = os.path.join(VOICES_DIR, f"{stem}.wav")
        txt_path = os.path.join(VOICES_DIR, f"{stem}.txt")

        if not os.path.isfile(wav_path):
            for fn in os.listdir(VOICES_DIR):
                if fn.lower().startswith(stem) and fn.lower().endswith(".wav"):
                    wav_path = os.path.join(VOICES_DIR, fn)
                    break

        if not os.path.isfile(txt_path):
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(spec["ref_text"] + "\n")

        ref_txt_val = spec["ref_text"]
        if os.path.isfile(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    read_t = f.read().strip()
                if read_t:
                    ref_txt_val = read_t
            except Exception:
                pass

        if (
            role not in profiles
            or profiles[role].get("ref_audio_path") != wav_path
            or profiles[role].get("ref_text") != ref_txt_val
        ):
            profiles[role] = {
                "ref_audio_path": wav_path,
                "ref_text": ref_txt_val,
                "speed": spec["speed"],
            }
            changed = True

    if changed or not os.path.isfile(PROFILES_JSON):
        with open(PROFILES_JSON, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=4)

    return profiles


def _normalize_for_f5(text: str) -> str:
    s = re.sub(r"§[0-9a-fk-or]", "", text)
    s = s.replace("**", "").replace("`", "").replace("➔", " to ").replace("->", " to ")
    s = re.sub(r"\(\s*ΔX=[^)]+\)", "", s)
    s = s.replace("3x3", "3 by 3").replace("EU/t", "E U per tick")
    s = re.sub(r"\b([XZ])=(-?\d+)", lambda m: f"{m.group(1)} {'minus ' + m.group(2)[1:] if m.group(2).startswith('-') else m.group(2)}", s)
    s = re.sub(r"(?<!\w)-(\d+)", r"minus \1", s)
    s = s.replace("°", " degrees")
    compass_map = {
        r"\bN\b": "North", r"\bNE\b": "Northeast", r"\bE\b": "East", r"\bSE\b": "Southeast",
        r"\bS\b": "South", r"\bSW\b": "Southwest", r"\bW\b": "West", r"\bNW\b": "Northwest",
    }
    for pat, word in compass_map.items():
        s = re.sub(pat, word, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class PitCrewVoiceEngine:
    def __init__(
        self,
        log_callback=None,
        ingame_chat_callback=None,
        stt_ready_callback=None,
        token_audit_callback=None,
        task_hud_toggle_callback=None,
    ):
        self.log_cb = log_callback
        self.chat_cb = ingame_chat_callback
        self.stt_cb = stt_ready_callback
        self.token_cb = token_audit_callback
        self.task_hud_cb = task_hud_toggle_callback

        self.tts_enabled = True
        self.narrator_voice_enabled = False
        self.ingame_text_enabled = False
        self.proactive_enabled = True
        self.engine_mode = "PRODUCTION"
        self.active_task_title = "Mine 500 Copper & Cassiterite Ore"
        self.active_task_category = "⛏️ Mining & Extraction"
        self.active_lead_specialist = "Geologist / Navigator"
        self.active_task_filter = "copper, chalcopyrite, tetrahedrite, tin, cassiterite"

        self.ptt_mode = "~ (Grave) / Mouse4"
        self.task_hud_key_name = "J Key (Task Menu)"

        self.voice_profiles = ensure_voice_profiles()
        self._tts_queue = queue.Queue()
        self._f5_proc = None

        self._last_target_chunk = None
        self._announced_50m = False
        self._announced_center = False
        self._last_proactive_ts = 0.0
        self._last_wrong_way_ts = 0.0
        self._last_hazard_ts = 0.0
        self._last_health = 20.0
        self._last_combat_ts = 0.0

        self._dist_history = deque(maxlen=16)

        self._whisper_model = None
        self._ptt_recording = False
        self._task_key_was_down = False

        threading.Thread(target=self._tts_worker_loop, daemon=True, name="Atlas-F5TTS-Worker").start()
        threading.Thread(target=self._ptt_poller_loop, daemon=True, name="Atlas-PTT-Poller").start()

    def _send_hud_text(self, msg: str):
        if self.ingame_text_enabled and self.chat_cb and msg:
            self.chat_cb(msg)

    def is_f5_online(self) -> bool:
        try:
            r = requests.get(F5_HEALTH_URL, timeout=0.6)
            return r.status_code == 200
        except Exception:
            return False

    def boot_f5_server(self):
        if self.is_f5_online():
            if self.log_cb:
                self.log_cb("READY", "F5-TTS Audio Cortex is already ONLINE on Port 1237.")
            return
        threading.Thread(target=self._spawn_f5_thread, daemon=True).start()

    def _find_f5_python_exe(self) -> str:
        preferred = r"E:\SovereignEngine\AA_Dynamics_Engine\audio_cortex\f5-env\Scripts\python.exe"
        if os.path.isfile(preferred):
            return preferred
        candidates = [sys.executable]
        root = r"E:\SovereignEngine"
        if os.path.isdir(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if "python.exe" in [f.lower() for f in filenames]:
                    py_exe = os.path.join(dirpath, "python.exe")
                    if py_exe not in candidates:
                        candidates.append(py_exe)
                if dirpath.count(os.sep) - root.count(os.sep) > 3:
                    del dirnames[:]
        for py_exe in candidates:
            try:
                res = subprocess.run(
                    [py_exe, "-c", "import f5_tts, fastapi, uvicorn; print('F5_OK')"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                if "F5_OK" in (res.stdout or ""):
                    return py_exe
            except Exception:
                continue
        return sys.executable

    def _spawn_f5_thread(self):
        py_exe = self._find_f5_python_exe()
        server_script = os.path.join(ENGINE_ROOT, "tts_server.py")
        if self.log_cb:
            self.log_cb("SYSTEM", f"Booting F5-TTS Audio Cortex on Port 1237 via {py_exe}...")
        try:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            self._f5_proc = subprocess.Popen(
                [py_exe, server_script],
                cwd=ENGINE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            for line in self._f5_proc.stdout:
                clean = line.strip()
                if clean and any(k in clean.lower() for k in ("booting", "online", "synthesizing", "error", "uvicorn running", "no module")):
                    if self.log_cb:
                        tag = "READY" if "online" in clean.lower() else ("ERROR" if "error" in clean.lower() or "no module" in clean.lower() else "ENGINE")
                        self.log_cb(tag, f"[:1237] {clean}")
        except Exception as e:
            if self.log_cb:
                self.log_cb("ERROR", f"Failed to boot tts_server.py on Port 1237: {e}")

    def stop_f5_server(self):
        if self._f5_proc and self._f5_proc.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self._f5_proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                self._f5_proc.terminate()

    def speak(self, role: str, text: str, force_narrator: bool = False):
        if not self.tts_enabled:
            return
        if role == "Story Chronicler" and not (self.narrator_voice_enabled or force_narrator):
            return

        clean = _normalize_for_f5(text)
        if not clean:
            return

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
        if not sentences:
            return
        short_speech = sentences[0]
        if len(sentences) > 1 and (len(short_speech) + len(sentences[1])) <= 155:
            short_speech = f"{short_speech} {sentences[1]}"
        if len(short_speech) > 175:
            short_speech = short_speech[:172].rsplit(" ", 1)[0] + "."

        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break
        self._tts_queue.put((role, short_speech))

    def _tts_worker_loop(self):
        while True:
            role, text = self._tts_queue.get()
            if not self.tts_enabled or not text:
                continue

            self.voice_profiles = ensure_voice_profiles()
            prof = self.voice_profiles.get(role) or self.voice_profiles.get("Geologist / Navigator")
            played = False

            try:
                payload = {
                    "gen_text": text,
                    "ref_audio_path": prof["ref_audio_path"],
                    "ref_text": prof["ref_text"],
                    "speed": float(prof.get("speed", 1.0)),
                }
                resp = requests.post(F5_TTS_URL, json=payload, timeout=(2.5, 45.0))
                if resp.status_code == 200 and resp.content:
                    if self.token_cb:
                        self.token_cb("F5-TTS :1237", role, text)
                    data_out, sr_out = sf.read(io.BytesIO(resp.content), dtype="float32")
                    try:
                        import sounddevice as sd
                        sd.play(data_out, sr_out)
                        sd.wait()
                    except Exception:
                        tmp_out = os.path.join(tempfile.gettempdir(), "atlas_f5_out.wav")
                        sf.write(tmp_out, data_out, sr_out, format="WAV", subtype="PCM_16")
                        winsound.PlaySound(tmp_out, winsound.SND_FILENAME)
                    played = True
                elif self.log_cb:
                    self.log_cb("WARN", f"F5-TTS HTTP {resp.status_code}: {resp.text[:120]}")
            except Exception:
                played = False

            if not played and os.name == "nt":
                try:
                    safe_txt = text.replace('"', "'").replace("\n", " ")
                    ps_cmd = (
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        "$s.Rate = 1; "
                        f'$s.Speak("{safe_txt}");'
                    )
                    subprocess.run(
                        ["powershell", "-NoProfile", "-Command", ps_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                except Exception:
                    pass

    def _matches_active_task(self, text_to_check: str) -> bool:
        task_raw = (self.active_task_filter or "").strip().lower()
        if not task_raw or task_raw in ("all", "*", "any"):
            return True
        tokens = [t.strip() for t in re.split(r"[,;/\s]+", task_raw) if len(t.strip()) >= 3]
        if not tokens:
            return True
        haystack = (text_to_check or "").lower()
        return any(tok in haystack for tok in tokens)

    def _find_best_task_target_vein(self, telemetry: dict, ore: dict) -> dict:
        """Finds the nearest discovered vein matching the active task, or falls back to the nearest 3x3 ore center."""
        bx = float(telemetry.get("x", 0.0))
        bz = float(telemetry.get("z", 0.0))
        summary = telemetry.get("discovered_veins_summary") or {}
        for v in summary.get("nearest_veins", []):
            desc = f"{v.get('name', '')} {v.get('ores', '')}"
            if self._matches_active_task(desc):
                return {
                    "name": v.get("name", "target ore"),
                    "x": int(v.get("x", ore.get("target_block_x", 0))),
                    "z": int(v.get("z", ore.get("target_block_z", 0))),
                    "distance": float(v.get("distance_blocks", ore.get("distance_blocks", 99.0))),
                    "compass": re.sub(r"\s*\([^)]*\)", "", str(v.get("compass", ore.get("compass", "ahead")))),
                    "y_range": v.get("y_range", "40-80"),
                    "is_known_vein": True,
                }
        return {
            "name": self.active_task_title,
            "x": int(ore.get("target_block_x", 0)),
            "z": int(ore.get("target_block_z", 0)),
            "distance": float(ore.get("distance_blocks", 99.0)),
            "compass": re.sub(r"\s*\([^)]*\)", "", str(ore.get("compass", "ahead"))),
            "y_range": "20-70",
            "is_known_vein": False,
        }

    def evaluate_proactive_tripwires(self, telemetry: dict, ore: dict, hazards: list):
        if not self.proactive_enabled:
            return
        now = time.time()

        # 1. Base Explosion / Voltage Hazard Always Has Priority
        if hazards and (now - self._last_hazard_ts) > 90.0:
            self._last_hazard_ts = now
            h_msg = hazards[0].get("message", "Critical base hazard detected!")
            callout = f"Hazard alert! {h_msg}"
            self._send_hud_text(f"§c[SAFETY OFFICER] §f{callout}")
            self.speak("Safety & Voltage Officer", callout)
            return

        # 2. Critical Low-HP Emergency (Only if not locked out by strict Mining Production mode, or if HP <= 6.0)
        cur_hp = float(telemetry.get("health", 20.0))
        prev_hp = getattr(self, "_last_health", 20.0)
        self._last_health = cur_hp
        hp_drop = prev_hp - cur_hp
        if (cur_hp <= 6.0 and cur_hp < prev_hp) and (now - self._last_combat_ts) > 90.0:
            self._last_combat_ts = now
            hearts = round(cur_hp / 2.0, 1)
            c_msg = f"Critical health at {hearts} hearts! Barricade or sprint-jump out now."
            self.speak("Combat Specialist", c_msg)
            return

        # 3. PRODUCTION MODE: WRONG-WAY VECTOR ENFORCER & GEOLOGIST-ONLY LOCK
        target_info = self._find_best_task_target_vein(telemetry, ore)
        bx = float(telemetry.get("x", 0.0))
        bz = float(telemetry.get("z", 0.0))
        cur_dist = math.hypot(target_info["x"] - bx, target_info["z"] - bz)

        self._dist_history.append((now, bx, bz, cur_dist, target_info["name"], target_info["compass"]))

        if self.engine_mode == "PRODUCTION" and "Mining" in self.active_task_category:
            # Check if player has been moving the WRONG WAY (increasing distance to target vein over last 6 seconds)
            old_samples = [s for s in self._dist_history if 4.5 <= (now - s[0]) <= 9.0 and s[4] == target_info["name"]]
            if old_samples and (now - self._last_wrong_way_ts) > 65.0:
                old_t, old_x, old_z, old_dist, _, _ = old_samples[0]
                moved_dist = math.hypot(bx - old_x, bz - old_z)
                dist_delta = cur_dist - old_dist
                # Player moved at least 11 blocks, and at least 8.5 of those blocks were AWAY from the target vein!
                if moved_dist >= 11.0 and dist_delta >= 8.5 and cur_dist > 22.0:
                    self._last_wrong_way_ts = now
                    self._dist_history.clear()
                    w_msg = (
                        f"Hey, you're not going to make up time going the wrong way! "
                        f"For {self.active_task_title}, bear {target_info['compass']} to X {target_info['x']}, Z {target_info['z']}."
                    )
                    if self.log_cb:
                        self.log_cb("WARN", f"[EFFICIENCY ENFORCER] {w_msg}")
                    self.speak("Geologist / Navigator", w_msg)
                    return

        # 4. Task-Filtered Geologist Proximity Callout
        if not target_info["is_known_vein"] and self.engine_mode != "PRODUCTION":
            return

        target_chunk = (target_info["x"] >> 4, target_info["z"] >> 4)
        in_center = (int(bx) >> 4 == target_chunk[0]) and (int(bz) >> 4 == target_chunk[1])
        speed_bps = float(telemetry.get("player_speed_bps", 0.0))

        if target_chunk != self._last_target_chunk:
            self._last_target_chunk = target_chunk
            self._announced_50m = False
            self._announced_center = False

        if in_center and not self._announced_center and (now - self._last_proactive_ts) > 90.0:
            self._announced_center = True
            self._announced_50m = True
            self._last_proactive_ts = now
            msg = f"On task: inside the {target_info['name']} chunk. Dig to Y {target_info['y_range']}."
            if self.log_cb:
                self.log_cb("READY", f"[TASK VEIN LOCK] {msg}")
            self.speak("Geologist / Navigator", msg)
            return

        if (18.0 <= cur_dist <= 45.0) and not in_center and not self._announced_50m and speed_bps >= 1.5 and (now - self._last_proactive_ts) > 120.0:
            self._announced_50m = True
            self._last_proactive_ts = now
            msg = f"Task target {target_info['name']} is {int(round(cur_dist))} blocks {target_info['compass']} at Y {target_info['y_range']}."
            if self.log_cb:
                self.log_cb("SYSTEM", f"[TASK ETA] {msg}")
            self.speak("Geologist / Navigator", msg)

    def _is_ptt_held(self) -> bool:
        if os.name != "nt":
            return False
        vks = PTT_KEYS.get(self.ptt_mode, (0xC0,))
        user32 = ctypes.windll.user32
        for vk in vks:
            if user32.GetAsyncKeyState(vk) & 0x8000:
                return True
        return False

    def _is_task_hud_key_pressed(self) -> bool:
        if os.name != "nt":
            return False
        vk = TASK_HUD_KEYS.get(self.task_hud_key_name, 0x4A)
        user32 = ctypes.windll.user32
        # Only trigger if not typing into an active text chat or if F6 is used
        return bool(user32.GetAsyncKeyState(vk) & 0x8000)

    def _ptt_poller_loop(self):
        while True:
            if self._is_ptt_held() and not self._ptt_recording:
                self._record_and_transcribe()

            hud_down = self._is_task_hud_key_pressed()
            if hud_down and not self._task_key_was_down:
                self._task_key_was_down = True
                if self.task_hud_cb:
                    self.task_hud_cb()
            elif not hud_down:
                self._task_key_was_down = False

            time.sleep(0.06)

    def _record_and_transcribe(self):
        self._ptt_recording = True
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            time.sleep(1.0)
            self._ptt_recording = False
            return

        samplerate = 16000
        chunks = []

        if self.log_cb:
            self.log_cb("SYSTEM", "🎙️ Push-to-Talk active — recording microphone...")

        def _callback(indata, frames, time_info, status):
            chunks.append(indata.copy())

        t0 = time.time()
        try:
            with sd.InputStream(samplerate=samplerate, channels=1, dtype="int16", callback=_callback):
                while self._is_ptt_held() and (time.time() - t0) < 20.0:
                    time.sleep(0.05)
        except Exception as e:
            if self.log_cb:
                self.log_cb("ERROR", f"Microphone capture error: {e}")
            self._ptt_recording = False
            return

        if (time.time() - t0) < 0.35 or not chunks:
            self._ptt_recording = False
            return

        audio_np = np.concatenate(chunks, axis=0)
        tmp_wav = os.path.join(tempfile.gettempdir(), "atlas_ptt_input.wav")
        try:
            with wave.open(tmp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(samplerate)
                wf.writeframes(audio_np.tobytes())

            if self._whisper_model is None:
                if self.log_cb:
                    self.log_cb("SYSTEM", "Loading faster-whisper (base.en, CPU int8 — 0 MB VRAM)...")
                from faster_whisper import WhisperModel
                self._whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

            segments, _ = self._whisper_model.transcribe(tmp_wav, beam_size=3, vad_filter=True)
            collected_parts = []
            for seg_item in segments:
                collected_parts.append(seg_item.text.strip())
            transcript = " ".join(collected_parts).strip()

            if transcript:
                if self.token_cb:
                    self.token_cb("Whisper PTT", "Microphone STT", transcript)
                if self.log_cb:
                    self.log_cb("READY", f"🎙️ Transcribed: \"{transcript}\"")
                if self.stt_cb:
                    self.stt_cb(transcript)
        except Exception as e:
            if self.log_cb:
                self.log_cb("ERROR", f"Push-to-Talk STT error: {e}")
        finally:
            if os.path.isfile(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
            self._ptt_recording = False