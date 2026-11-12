import os
import re
import sys
import time
import json
import socket
import threading
import subprocess
from datetime import datetime
import requests
from PySide6 import QtCore, QtGui, QtWidgets

ENGINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from core.gtnh_math_core import (
    GT_VOLTAGE_TIERS,
    resolve_nearest_ore_center,
    calculate_gt_overclock,
    evaluate_safety_hazards,
)
from core.prospecting_cache_reader import ProspectingCacheIndex
from core.voice_comms import PitCrewVoiceEngine, PTT_KEYS, TASK_HUD_KEYS
from core.task_manager import TaskManager, TaskOverlayHUD
from core.story_chronicler import StoryChronicler, STORY_THEMES
from core.hq_token_logger import HQTokenLogger

MODELS_DIR = r"E:\SovereignEngine\AA_Dynamics_Training_Models"
LORA_DIR = r"E:\SovereignEngine\AA_Dynamics_Training_Facility\LoRA_adapters"
CUDA_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
CONFIG_PATH = os.path.join(ENGINE_ROOT, "config", "engine_config.json")


def load_config() -> dict:
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "telemetry_host": "127.0.0.1",
        "telemetry_tcp_port": 8765,
        "llm_host": "127.0.0.1",
        "llm_port": 1234,
    }


def find_llama_server():
    candidates = [
        r"E:\SovereignEngine\AA_Dynamics_Engine\llama-server.exe",
        r"E:\SovereignEngine\AA_Dynamics_Training_Models\llama-server.exe",
        os.path.join(ENGINE_ROOT, "llama-server.exe"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    sovereign_root = r"E:\SovereignEngine"
    if os.path.isdir(sovereign_root):
        for entry in os.listdir(sovereign_root):
            direct = os.path.join(sovereign_root, entry, "llama-server.exe")
            if os.path.isfile(direct):
                return direct
    return None


def scan_training_models():
    if not os.path.isdir(MODELS_DIR):
        return []
    models = [
        f for f in os.listdir(MODELS_DIR)
        if f.lower().endswith(".gguf") and not f.lower().startswith("mmproj")
    ]

    def sort_key(name: str):
        low = name.lower()
        if "deepseek-coder-v2-lite" in low:
            return (0, low)
        if "llama" in low or "qwen2.5-coder" in low:
            return (1, low)
        return (2, low)

    models.sort(key=sort_key)
    return models


def discover_lora_sidecars():
    if not os.path.isdir(LORA_DIR):
        return []
    lora_files = []
    for root_dir, dirs, files in os.walk(LORA_DIR):
        dirs.sort()
        files_sorted = sorted(files)
        if os.path.abspath(root_dir) == os.path.abspath(LORA_DIR):
            for f in files_sorted:
                if f.lower().endswith(".gguf"):
                    lora_files.append(os.path.join(root_dir, f))
        else:
            folder = os.path.basename(root_dir)
            ggufs = [f for f in files_sorted if f.lower().endswith(".gguf")]
            if ggufs:
                pref = f"{folder.lower()}.gguf"
                chosen = next((f for f in ggufs if f.lower() == pref), ggufs[0])
                lora_files.append(os.path.join(root_dir, chosen))
    return lora_files


def extract_model_signature(name_str: str):
    s = os.path.basename(name_str.rstrip("\\/")).lower()
    if s.endswith(".gguf"):
        s = s[:-5]
    family = None
    for fam in (
        "qwen2.5", "qwen3", "qwen2",
        "deepseek-coder-v2", "deepseek-r1", "deepseek",
        "llama-3.3", "llama-3.2", "llama-3.1", "llama-3",
        "mistral", "mixtral", "codestral",
        "gemma-2", "gemma", "phi-4", "phi-3",
    ):
        if fam in s:
            family = fam
            break
    size_m = re.search(r"(?:^|[-_])(\d+b|lite)(?:$|[-_])", s)
    size = size_m.group(1) if size_m else None
    is_vl = "-vl-" in s or s.endswith("-vl")
    return family, size, is_vl


def filter_compatible_sidecars(model_filename: str, sidecar_paths: list):
    model_sig = extract_model_signature(model_filename)
    compatible, skipped = [], []

    for lf in sidecar_paths:
        parent_dir = os.path.dirname(lf)
        cfg_file = os.path.join(parent_dir, "adapter_config.json")
        label = (
            os.path.basename(parent_dir)
            if os.path.abspath(parent_dir) != os.path.abspath(LORA_DIR)
            else os.path.basename(lf)
        )
        if not os.path.isfile(cfg_file) or model_sig[0] is None:
            compatible.append(lf)
            continue
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg_base = str(json.load(f).get("base_model_name_or_path") or "").strip()
        except Exception:
            compatible.append(lf)
            continue
        if not cfg_base:
            compatible.append(lf)
            continue
        lora_sig = extract_model_signature(cfg_base)
        if lora_sig[0] is None or lora_sig == model_sig:
            compatible.append(lf)
        else:
            skipped.append(label)

    return compatible, skipped


class StandaloneEngineManager(QtCore.QObject):
    statusChanged = QtCore.Signal(str, str, str)
    engineLog = QtCore.Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_engines = {}
        self._lock = threading.Lock()

    def is_port_active(self, port) -> bool:
        with self._lock:
            info = self.active_engines.get(str(port))
            return bool(info and info["proc"].poll() is None)

    def get_port_summary(self, port):
        port = str(port)
        with self._lock:
            info = self.active_engines.get(port)
            if not info or info["proc"].poll() is not None:
                return "OFFLINE", f"PORT {port}: OFFLINE"
            return info["state"], info.get("summary", f"PORT {port}: {info['state']}")

    def load_model(self, port, model_filename, context_size):
        port = str(port).strip()
        model_path = os.path.join(MODELS_DIR, model_filename or "")
        exe_path = find_llama_server()

        if not model_filename or model_filename == "NO MODELS FOUND" or not os.path.isfile(model_path):
            self.statusChanged.emit(port, "ERROR", "ERROR: Invalid model")
            self.engineLog.emit("ERROR", f"Model file not found: {model_path}")
            return
        if not exe_path:
            self.statusChanged.emit(port, "ERROR", "ERROR: Missing llama-server.exe")
            self.engineLog.emit("ERROR", "Could not locate llama-server.exe across E:\\SovereignEngine.")
            return

        if self.is_port_active(port):
            self.unload_model(port)

        threading.Thread(
            target=self._boot_thread,
            args=(exe_path, model_path, model_filename, str(context_size), port),
            daemon=True,
        ).start()

    def _zero_loras(self, port):
        try:
            url = f"http://127.0.0.1:{port}/lora-adapters"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                adapters = r.json()
                if adapters:
                    requests.post(url, json=[{"id": a["id"], "scale": 0.0} for a in adapters], timeout=5)
                return len(adapters)
        except Exception:
            pass
        return 0

    def _emit_filtered_cpp_line(self, port, clean_line):
        if re.fullmatch(r"[.\s]+", clean_line):
            return
        low = clean_line.lower()
        if any(k in low for k in ("error", "failed", "fatal", "out of memory", "cuda error", "invalid argument", "exception")):
            self.engineLog.emit("ERROR", f"[:{port}] {clean_line}")
        elif any(k in low for k in (
            "ggml_cuda_init", "device ", "load_tensors", "offload",
            "buffer size", "lora", "n_ctx", "listening", "tokens per second",
        )):
            self.engineLog.emit("ENGINE", f"[:{port}] {clean_line}")

    def _boot_thread(self, exe_path, model_path, model_filename, context_size, port):
        with self._lock:
            any_running = any(v["proc"].poll() is None for v in self.active_engines.values())
        if not any_running and os.name == "nt":
            subprocess.run(
                "taskkill /F /IM llama-server.exe /T",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.8)

        short_m = model_filename[:-5] if model_filename.lower().endswith(".gguf") else model_filename
        short_m = short_m[:24] + ".." if len(short_m) > 26 else short_m

        exe_dir = os.path.dirname(os.path.abspath(exe_path))
        env = os.environ.copy()
        path_prefixes = [exe_dir]
        if os.path.isdir(CUDA_PATH):
            path_prefixes.append(CUDA_PATH)
        env["PATH"] = os.pathsep.join(path_prefixes) + os.pathsep + env.get("PATH", "")

        base_cmd = [
            exe_path, "-m", model_path, "-c", context_size,
            "-ngl", "999", "-np", "1", "--host", "127.0.0.1", "--port", port,
        ]
        if "-VL-" in model_filename.upper():
            mmproj = os.path.join(MODELS_DIR, "mmproj-Qwen2.5-VL-72B-Instruct-f32.gguf")
            if os.path.isfile(mmproj):
                base_cmd.extend(["--mmproj", mmproj])

        all_sidecars = discover_lora_sidecars()
        compat_sidecars, skipped_names = filter_compatible_sidecars(model_filename, all_sidecars)
        if skipped_names:
            self.engineLog.emit(
                "WARN",
                f"Skipped {len(skipped_names)} non-matching LoRA sidecar(s) ({', '.join(skipped_names)}).",
            )

        attempts = []
        if compat_sidecars:
            comma_cmd = list(base_cmd) + ["--lora-init-without-apply", "--lora", ",".join(compat_sidecars)]
            attempts.append(("WITH_LORAS_COMMA", comma_cmd, len(compat_sidecars)))
            if len(compat_sidecars) > 1:
                rep_cmd = list(base_cmd) + ["--lora-init-without-apply"]
                for lf in compat_sidecars:
                    rep_cmd.extend(["--lora", lf])
                attempts.append(("WITH_LORAS_LEGACY", rep_cmd, len(compat_sidecars)))
        attempts.append(("BASE_ONLY", base_cmd, 0))

        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        for mode, cmd, exp_loras in attempts:
            init_msg = f"PORT {port}: LOADING {short_m} (0s)..."
            self.statusChanged.emit(port, "LOADING", init_msg)
            self.engineLog.emit(
                "SYSTEM",
                f"Spawning llama-server.exe on Port {port} | Model: {model_filename} | CTX: {context_size} | LoRAs: {exp_loras}",
            )

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=exe_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    env=env,
                    creationflags=flags,
                )
            except Exception as e:
                self.statusChanged.emit(port, "ERROR", f"PORT {port}: BOOT ERROR")
                self.engineLog.emit("ERROR", f"Failed to spawn llama-server.exe: {e}")
                return

            with self._lock:
                self.active_engines[port] = {
                    "proc": proc,
                    "model": model_filename,
                    "state": "LOADING",
                    "summary": init_msg,
                }

            ready_event = threading.Event()
            boot_t0 = time.time()

            def mark_ready():
                if ready_event.is_set():
                    return
                ready_event.set()
                staged = self._zero_loras(port)
                tag = f"{staged} LoRAs Ready" if staged > 0 else "Base Weights"
                msg = f"ONLINE (:{port}) | {short_m} [{tag}]"
                with self._lock:
                    if port in self.active_engines and self.active_engines[port]["proc"] == proc:
                        self.active_engines[port]["state"] = "READY"
                        self.active_engines[port]["summary"] = msg
                elapsed = time.time() - boot_t0
                self.statusChanged.emit(port, "READY", msg)
                self.engineLog.emit("READY", f"{msg} (loaded in {elapsed:.1f}s)")

            def health_poller():
                last_sec = -1
                health_url = f"http://127.0.0.1:{port}/health"
                while proc.poll() is None and not ready_event.is_set():
                    elapsed_s = int(time.time() - boot_t0)
                    if elapsed_s != last_sec and not ready_event.is_set():
                        last_sec = elapsed_s
                        tick_msg = f"PORT {port}: LOADING {short_m} ({elapsed_s}s)..."
                        with self._lock:
                            if port in self.active_engines and self.active_engines[port]["proc"] == proc:
                                self.active_engines[port]["summary"] = tick_msg
                        self.statusChanged.emit(port, "LOADING", tick_msg)
                    try:
                        r = requests.get(health_url, timeout=0.6)
                        if r.status_code == 200:
                            mark_ready()
                            return
                    except Exception:
                        pass
                    time.sleep(0.5)

            threading.Thread(target=health_poller, daemon=True).start()

            fd = proc.stdout.fileno()
            buf = ""
            while True:
                try:
                    chunk = os.read(fd, 2048)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf or "\r" in buf:
                    idx_n = buf.find("\n")
                    idx_r = buf.find("\r")
                    cut = idx_r if idx_n == -1 else (idx_n if idx_r == -1 else min(idx_n, idx_r))
                    line_part = buf[:cut].strip()
                    buf = buf[cut + 1:]
                    if not line_part:
                        continue
                    low = line_part.lower()
                    if "listening on http" in low or "server listening at" in low or "http server listening" in low:
                        self._emit_filtered_cpp_line(port, line_part)
                        mark_ready()
                    else:
                        self._emit_filtered_cpp_line(port, line_part)

            if buf.strip():
                self._emit_filtered_cpp_line(port, buf.strip())

            proc.wait()

            with self._lock:
                cur = self.active_engines.get(port)
                manual_stop = not cur or cur["proc"] != proc
            if manual_stop:
                return

            if not ready_event.is_set() and mode != "BASE_ONLY":
                self.engineLog.emit("WARN", f"Mode {mode} exited early on Port {port}; retrying fallback...")
                continue

            with self._lock:
                if port in self.active_engines and self.active_engines[port]["proc"] == proc:
                    del self.active_engines[port]

            if not ready_event.is_set():
                self.statusChanged.emit(port, "ERROR", f"PORT {port}: BOOT FAILED")
                self.engineLog.emit("ERROR", f"llama-server.exe exited before ready (exit={proc.returncode}).")
            else:
                self.statusChanged.emit(port, "OFFLINE", f"PORT {port}: OFFLINE")
                self.engineLog.emit("WARN", f"Engine on Port {port} stopped.")
            return

    def unload_model(self, port):
        port = str(port).strip()
        with self._lock:
            entry = self.active_engines.pop(port, None)
        if entry and entry["proc"].poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(entry["proc"].pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                entry["proc"].terminate()
            self.engineLog.emit("SYSTEM", f"Unloaded engine & cleared VRAM on Port {port}.")
        elif os.name == "nt" and port == "1234":
            subprocess.run(
                "taskkill /F /IM llama-server.exe /T",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.engineLog.emit("SYSTEM", f"Cleared external llama-server.exe on Port {port}.")
        self.statusChanged.emit(port, "OFFLINE", f"PORT {port}: OFFLINE")

    def unload_all(self):
        with self._lock:
            ports = list(self.active_engines.keys())
        for p in ports:
            self.unload_model(p)


CREW_ROSTER = {
    "Geologist / Navigator": (
        "You are the GTNH 2.9.0 Chief Geologist & Subsurface Ore Specialist aboard the Sovereign Pit Crew. "
        "NEVER guess chunk math—rely strictly on the pre-computed Ore-Grid Telemetry and the ServerCache / VisualProspecting veins injected below. "
        "Give crisp, actionable mining directions (compass bearing, exact block coordinates, discovered vein status, and Y-level vein tips for GTNH 2.9.0) in 3-5 short sentences."
    ),
    "Wayfinder / Map Specialist": (
        "You are the GTNH 2.9.0 Chief Wayfinder, Cartographer & Expedition Scout. "
        "You specialize in surface navigation, JourneyMap waypoint routing, biome analysis (IC2 crop humidity/nutrient properties, Forestry bee climates, water tank generation, RWG2 terrain), and safe return-to-base vectors. "
        "Use the player's live coordinates, player_speed_bps movement speed, current biome, and the nearest_waypoints list from JourneyMap to give exact compass headings, travel ETAs, and multi-stop surface routes in 3-5 short sentences."
    ),
    "Chief Industrial Planner": (
        "You are the GTNH 2.9.0 Chief Industrial & Critical-Path Planner. "
        "Focus on batch-crafting efficiency, parallel machine scheduling (EBF, Bender, Wiremill, Chemical Reactor), "
        "and EU/t power budgeting using the deterministic telemetry and discovered ore reserves provided. Keep callouts concise and tactical."
    ),
    "Weapon / Armor / Tool Specialist": (
        "You are the GTNH 2.9.0 Master Armorer & Toolsmith. "
        "Guide the player in crafting and upgrading the optimal Tinkers' Construct tools, crossbows, weapons, and GregTech/IC2 armor for their EXACT current progression stage. "
        "CRITICAL RULE: Check the player's telemetry (loaded GT machines, held_item, and discovered ores) and NEVER recommend gated materials they cannot smelt or alloy yet "
        "(e.g., do not suggest Steel, Alumite, or Cobalt if they have 0 GT machines or are in the Stone/Steam age; focus on Iguana Tinker Tweaks tool XP leveling, part swapping, Flint -> Copper -> Bronze -> Iron -> Wrought Iron, and early survival armor)."
    ),
    "Combat Specialist": (
        "You are the GTNH 2.9.0 Combat & Survival Tactician. "
        "Analyze the player's live combat telemetry (health, food_level, player_speed_bps movement speed, held_item, crosshair_target, and biome) to coach their fighting and survival habits. "
        "Give sharp, practical tips on kiting speed, terrain barricading, weapon reach/knockback, Spice of Life nutrition, Hardcore Darkness lighting, and countering GTNH Infernal Mobs."
    ),
    "Safety & Voltage Officer": (
        "You are the GTNH 2.9.0 Safety & High-Voltage Auditor. "
        "Diagnose explosion hazards (uncovered machines in rain, over-voltage hull connections, cable amperage fires, dry steam boilers) "
        "and give immediate step-by-step containment instructions before the player loses their base."
    ),
    "Story Chronicler": (
        "You are the Royal Chronicler & Master World Narrator aboard the Sovereign Engine. "
        "Weave the player's live telemetry, biome, weather, structures, and actions into an unfolding literary novel."
    ),
}


class TelemetryTCPServer(QtCore.QThread):
    packetReceived = QtCore.Signal(dict)
    serverStatus = QtCore.Signal(str)

    def __init__(self, host="127.0.0.1", port=8765, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = int(port)
        self._running = True
        self._sock = None
        self._clients = []
        self._clients_lock = threading.Lock()

    def send_ingame_chat(self, chat_line: str):
        payload = (json.dumps({"chat_line": chat_line}, ensure_ascii=False) + "\n").encode("utf-8")
        with self._clients_lock:
            dead = []
            for c in self._clients:
                try:
                    c.sendall(payload)
                except OSError:
                    dead.append(c)
            for d in dead:
                self._clients.remove(d)

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)
        self._sock = srv
        try:
            srv.bind((self.host, self.port))
            srv.listen(4)
            self.serverStatus.emit(f"TCP TELEMETRY LISTENING ON {self.host}:{self.port}")
        except Exception as e:
            self.serverStatus.emit(f"TCP BIND ERROR ({self.port}): {e}")
            return

        while self._running:
            try:
                conn, addr = srv.accept()
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_client(self, conn: socket.socket, addr):
        self.serverStatus.emit(f"GTNH CLIENT CONNECTED ({addr[0]}:{addr[1]})")
        with self._clients_lock:
            self._clients.append(conn)
        buf = ""
        with conn:
            while self._running:
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buf += data.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                self.packetReceived.emit(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                except OSError:
                    break
        if self._running:
            self.serverStatus.emit(f"CLIENT DISCONNECTED — LISTENING ON {self.host}:{self.port}")


class CrewStreamWorker(QtCore.QThread):
    headerReady = QtCore.Signal(str, str)
    tokenDelta = QtCore.Signal(str)
    usageStats = QtCore.Signal(int, int)
    streamDone = QtCore.Signal()
    streamError = QtCore.Signal(str)

    def __init__(
        self,
        llm_host,
        llm_port,
        crew_role,
        telemetry_snapshot,
        ore_math,
        prospecting_summary,
        hazards,
        user_query,
        custom_sys_prompt=None,
        custom_temp=0.35,
        parent=None,
    ):
        super().__init__(parent)
        self.llm_host = llm_host
        self.llm_port = int(llm_port)
        self.crew_role = crew_role
        self.telemetry_snapshot = telemetry_snapshot
        self.ore_math = ore_math
        self.prospecting_summary = prospecting_summary
        self.hazards = hazards
        self.user_query = user_query
        self.custom_sys_prompt = custom_sys_prompt
        self.custom_temp = float(custom_temp)

    def run(self):
        sys_prompt = (self.custom_sys_prompt or CREW_ROSTER.get(self.crew_role, CREW_ROSTER["Geologist / Navigator"])) + "\nCRITICAL BREVITY RULE: Respond in MAXIMUM 1 TO 2 SHORT SENTENCES (under 30 words total). Cut all filler and focus strictly on the player's ACTIVE TASK."
        context_packet = (
            "=== LIVE GTNH 2.9.0 PRE-RENDER TELEMETRY ===\n"
            f"{json.dumps(self.telemetry_snapshot, indent=2, ensure_ascii=False)}\n\n"
            "=== DETERMINISTIC 3x3 ORE-GRID & COMPASS SOLVER ===\n"
            f"{json.dumps(self.ore_math, indent=2, ensure_ascii=False)}\n\n"
            "=== DISCOVERED ORE VEINS & WAYPOINTS (512M SERVER RADAR + JOURNEYMAP) ===\n"
            f"{json.dumps(self.prospecting_summary, indent=2, ensure_ascii=False)}\n\n"
            "=== ACTIVE DETERMINISTIC SAFETY ALERTS ===\n"
            f"{json.dumps(self.hazards, indent=2, ensure_ascii=False) if self.hazards else 'NONE'}\n\n"
            f"=== PLAYER REQUEST ===\n{self.user_query}"
        )

        self.headerReady.emit(self.crew_role, self.user_query)
        url = f"http://{self.llm_host}:{self.llm_port}/v1/chat/completions"
        payload = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": context_packet},
            ],
            "temperature": self.custom_temp,
            "top_p": 0.85,
            "min_p": 0.05,
            "repeat_penalty": 1.10,
            "repeat_last_n": 128,
            "dry_multiplier": 0.8,
            "max_tokens": 150,
            "cache_prompt": False,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        p_tok, c_tok = 0, 0
        gen_Accumulator = []
        try:
            with requests.post(url, json=payload, stream=True, timeout=(5, 120)) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data:"):
                        continue
                    data_str = raw[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        usage = chunk.get("usage")
                        if usage and usage.get("total_tokens"):
                            p_tok = int(usage.get("prompt_tokens", 0))
                            c_tok = int(usage.get("completion_tokens", 0))
                        choices = chunk.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta", {}).get("content", "")
                            if delta:
                                gen_Accumulator.append(delta)
                                self.tokenDelta.emit(delta)
                    except Exception:
                        continue
            if p_tok == 0 and c_tok == 0:
                p_tok = HQTokenLogger.estimate_tokens(sys_prompt + "\n" + context_packet)
                c_tok = HQTokenLogger.estimate_tokens("".join(gen_Accumulator))
            self.usageStats.emit(p_tok, c_tok)
            self.streamDone.emit()
        except Exception as e:
            self.streamError.emit(
                f"Port {self.llm_port} unreachable ({e}). Click [ LOAD MODEL ] in the top toolbar first."
            )


class BackgroundStoryWorker(QtCore.QThread):
    """Silently writes real-time behavioral & task-requisition chapters into the Markdown novel behind the scenes."""
    storyBeatReady = QtCore.Signal(str, str, int, int)

    def __init__(self, llm_host, llm_port, sys_prompt, temp, telemetry_snapshot, story_title, story_prompt, parent=None):
        super().__init__(parent)
        self.llm_host = llm_host
        self.llm_port = int(llm_port)
        self.sys_prompt = sys_prompt
        self.temp = float(temp)
        self.telemetry_snapshot = telemetry_snapshot
        self.story_title = story_title
        self.story_prompt = story_prompt

    def run(self):
        url = f"http://{self.llm_host}:{self.llm_port}/v1/chat/completions"
        user_msg = (
            f"=== LIVE WORLD STATE ===\n"
            f"Coords: ({self.telemetry_snapshot.get('x')}, {self.telemetry_snapshot.get('y')}, {self.telemetry_snapshot.get('z')}) | "
            f"Biome: {self.telemetry_snapshot.get('biome')} | Held: {self.telemetry_snapshot.get('held_item')} | "
            f"Target: {self.telemetry_snapshot.get('crosshair_target')} | HP: {self.telemetry_snapshot.get('health')}\n\n"
            f"=== NARRATIVE DIRECTIVE ===\n{self.story_prompt}"
        )
        payload = {
            "messages": [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temp,
            "top_p": 0.88,
            "max_tokens": 220,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                prose = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage") or {}
                p_tok = int(usage.get("prompt_tokens") or HQTokenLogger.estimate_tokens(self.sys_prompt + user_msg))
                c_tok = int(usage.get("completion_tokens") or HQTokenLogger.estimate_tokens(prose))
                if prose:
                    self.storyBeatReady.emit(self.story_title, prose, p_tok, c_tok)
        except Exception:
            pass


class PitCrewWindow(QtWidgets.QMainWindow):
    voiceQueryTranscribed = QtCore.Signal(str)
    toggleTaskOverlayRequested = QtCore.Signal()
    audioTokensLogged = QtCore.Signal()

    LOG_COLORS = {
        "SYSTEM": "#00D2FF",
        "ENGINE": "#94A3B8",
        "READY":  "#39FF14",
        "WARN":   "#FACC15",
        "ERROR":  "#FF003C",
    }

    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.llm_host = self.cfg.get("llm_host", "127.0.0.1")
        self.llm_port = int(self.cfg.get("llm_port", 1234))

        self.cache_index = ProspectingCacheIndex(self.cfg.get("gtnh_instance_dir"))
        self.chronicler = StoryChronicler()
        self.task_mgr = TaskManager()
        self._bg_story_worker = None
        self.hq_logger = HQTokenLogger()
        self._active_trigger_source = 'Cockpit UI'

        ico_path = os.path.join(ENGINE_ROOT, "voices", "icon.ico")
        if os.path.isfile(ico_path):
            self.setWindowIcon(QtGui.QIcon(ico_path))

        self.telemetry = {
            "x": -180,
            "y": 74,
            "z": 111,
            "player_speed_bps": 0.0,
            "health": 20.0,
            "food_level": 20,
            "dimension": "Overworld (0)",
            "biome": "Taiga",
            "held_item": "GregTech Prospecting Pick (LV)",
            "crosshair_target": "Grass Block",
            "is_raining": False,
            "is_thundering": False,
            "exposed_gt_machines": 0,
            "crosshair_machine_max_eu_t": 32,
            "crosshair_cable_eu_t": 32,
            "boiler_temp_c": 85.0,
            "boiler_water_mb": 4000,
        }
        self._sim_step = 0
        self._hazard_idx = 0
        self._active_worker = None
        self._mid_stream = False
        self._ingame_line_buf = ""
        self._tts_reply_buf = ""
        self._pending_book_reason = None
        self._last_dispatched_query = ""

        self.engine_mgr = StandaloneEngineManager(self)

        self.setWindowTitle("AA Dynamics — AI Cognitive Engine (v0.1.0-alpha)")
        self.resize(1420, 900)
        self.setStyleSheet("background-color: #060913; color: #E2E8F0; font-family: Consolas, 'Segoe UI', sans-serif;")

        self._build_engine_toolbar()
        self._build_ui()

        self.engine_mgr.statusChanged.connect(self._on_engine_status_changed)
        self.engine_mgr.engineLog.connect(self._append_engine_log)

        self.tcp_server = TelemetryTCPServer(
            self.cfg.get("telemetry_host", "127.0.0.1"),
            self.cfg.get("telemetry_tcp_port", 8765),
            self,
        )
        self.tcp_server.packetReceived.connect(self._on_external_telemetry)
        self.tcp_server.serverStatus.connect(lambda s: self.tcp_badge.setText(f"● {s}"))
        self.tcp_server.start()

        self.voiceQueryTranscribed.connect(self._on_voice_transcript)
        self.voice_engine = PitCrewVoiceEngine(
            log_callback=lambda cat, msg: QtCore.QMetaObject.invokeMethod(
                self, "_append_engine_log_slot", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, cat), QtCore.Q_ARG(str, msg)
            ),
            ingame_chat_callback=self.tcp_server.send_ingame_chat,
            stt_ready_callback=self.voiceQueryTranscribed.emit,
            token_audit_callback=self._on_audio_token_audit,
            task_hud_toggle_callback=self.toggleTaskOverlayRequested.emit,
        )
        self.task_overlay = TaskOverlayHUD(self.task_mgr)
        self.task_overlay.taskAddedByUser.connect(self._on_user_added_task)
        self.task_overlay.requestAiBreakdown.connect(self._on_request_ai_task_breakdown)
        self.task_overlay.activeTaskChanged.connect(self._sync_active_task_to_engine)
        self.toggleTaskOverlayRequested.connect(self._toggle_task_overlay_hud)
        self.audioTokensLogged.connect(self._refresh_token_hud_badge)

        self._sync_active_task_to_engine()
        self._refresh_telemetry_views()

        self.sim_timer = QtCore.QTimer(self)
        self.sim_timer.setInterval(450)
        self.sim_timer.timeout.connect(self._tick_walk_sim)

        self.model_timer = QtCore.QTimer(self)
        self.model_timer.setInterval(3000)
        self.model_timer.timeout.connect(self._poll_external_engine)
        self.model_timer.start()
        self._poll_external_engine()

        self._append_engine_log(
            "SYSTEM",
            "Sovereign Gaming Engine Online. Click [ 🧠 BOOT F5-TTS :1237 ] to load your 7 specialist voices into VRAM, and click [ LOAD MODEL ] (CTX: 32768).",
        )

    def _build_engine_toolbar(self):
        engine_bar = QtWidgets.QToolBar("Standalone Engine Control", self)
        engine_bar.setMovable(False)
        engine_bar.setFloatable(False)
        engine_bar.setStyleSheet(
            "QToolBar { background-color: #0B101D; border-bottom: 1px solid #00D2FF; spacing: 8px; padding: 5px 10px; } "
            "QLabel { color: #94A3B8; font-family: Consolas; font-size: 9.5pt; font-weight: bold; } "
            "QComboBox { background-color: #060913; color: #F8FAFC; border: 1px solid #334155; border-radius: 3px; padding: 4px 8px; font-family: Consolas; font-size: 9.5pt; } "
            "QComboBox:hover { border: 1px solid #00D2FF; } "
            "QComboBox QAbstractItemView { background-color: #0B101D; color: #F8FAFC; selection-background-color: #00D2FF; selection-color: #060913; }"
        )
        self.addToolBar(QtCore.Qt.TopToolBarArea, engine_bar)

        engine_bar.addWidget(QtWidgets.QLabel("MODEL:"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(330)
        found = scan_training_models()
        self.model_combo.addItems(found if found else ["NO MODELS FOUND"])
        engine_bar.addWidget(self.model_combo)

        rescan_btn = QtWidgets.QPushButton("↻")
        rescan_btn.setToolTip("Rescan AA_Dynamics_Training_Models")
        rescan_btn.setCursor(QtCore.Qt.PointingHandCursor)
        rescan_btn.setStyleSheet(
            "QPushButton { background-color: #060913; color: #00D2FF; border: 1px solid #334155; padding: 4px 8px; font-weight: bold; } "
            "QPushButton:hover { border: 1px solid #00D2FF; background-color: #1E293B; }"
        )
        rescan_btn.clicked.connect(self._rescan_models)
        engine_bar.addWidget(rescan_btn)

        engine_bar.addWidget(QtWidgets.QLabel("  CTX:"))
        self.ctx_combo = QtWidgets.QComboBox()
        self.ctx_combo.addItems(["8192", "16384", "32768", "65536", "131072"])
        self.ctx_combo.setCurrentText("32768")
        engine_bar.addWidget(self.ctx_combo)

        engine_bar.addWidget(QtWidgets.QLabel("  PORT:"))
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.addItems(["1234", "1235", "1236"])
        self.port_combo.setCurrentText(str(self.llm_port))
        self.port_combo.currentTextChanged.connect(self._on_port_combo_changed)
        engine_bar.addWidget(self.port_combo)

        self.load_btn = QtWidgets.QPushButton("[ LOAD MODEL ]")
        self.load_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.load_btn.setStyleSheet(
            "QPushButton { background-color: #060913; color: #39FF14; border: 1px solid #39FF14; padding: 5px 14px; font-family: Consolas; font-weight: bold; } "
            "QPushButton:hover { background-color: #39FF14; color: #060913; }"
        )
        self.load_btn.clicked.connect(
            lambda: self.engine_mgr.load_model(
                self.port_combo.currentText(), self.model_combo.currentText(), self.ctx_combo.currentText()
            )
        )
        engine_bar.addWidget(self.load_btn)

        self.unload_btn = QtWidgets.QPushButton("[ UNLOAD ]")
        self.unload_btn.setEnabled(False)
        self.unload_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.unload_btn.setStyleSheet(
            "QPushButton { background-color: #060913; color: #FF003C; border: 1px solid #FF003C; padding: 5px 12px; font-family: Consolas; font-weight: bold; } "
            "QPushButton:hover:enabled { background-color: #FF003C; color: #060913; } "
            "QPushButton:disabled { color: #475569; border: 1px solid #334155; }"
        )
        self.unload_btn.clicked.connect(lambda: self.engine_mgr.unload_model(self.port_combo.currentText()))
        engine_bar.addWidget(self.unload_btn)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        spacer.setStyleSheet("background: transparent;")
        engine_bar.addWidget(spacer)

        self.status_badge = QtWidgets.QLabel(f"PORT {self.llm_port}: OFFLINE")
        self.status_badge.setStyleSheet("color: #FF003C; font-family: Consolas; font-size: 10pt; font-weight: bold; padding-right: 6px;")
        engine_bar.addWidget(self.status_badge)

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(7)

        sub_bar = QtWidgets.QHBoxLayout()
        sub_bar.setSpacing(8)
        self.tcp_badge = QtWidgets.QLabel("● STARTING TCP :8765...")
        self.tcp_badge.setStyleSheet("color: #00D2FF; font-weight: bold; font-size: 9.5pt;")
        sub_bar.addWidget(self.tcp_badge)

        sub_bar.addStretch(1)

        self.f5_btn = QtWidgets.QPushButton("[ 🧠 BOOT F5-TTS :1237 ]")
        self.f5_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.f5_btn.setStyleSheet(
            "QPushButton { background-color: #1E1035; color: #E879F9; border: 1px solid #A855F7; padding: 4px 12px; font-weight: bold; } "
            "QPushButton:hover { background-color: #A855F7; color: #060913; }"
        )
        self.f5_btn.clicked.connect(lambda: self.voice_engine.boot_f5_server() if hasattr(self, "voice_engine") else None)
        sub_bar.addWidget(self.f5_btn)

        self.test_voice_btn = QtWidgets.QPushButton("[ 🔊 TEST VOICE ]")
        self.test_voice_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.test_voice_btn.setToolTip("Speak a test line using the currently selected Specialist's voice")
        self.test_voice_btn.setStyleSheet(
            "QPushButton { background-color: #0B101D; color: #38BDF8; border: 1px solid #38BDF8; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:hover { background-color: #38BDF8; color: #060913; }"
        )
        self.test_voice_btn.clicked.connect(self._test_active_specialist_voice)
        sub_bar.addWidget(self.test_voice_btn)

        self.tts_btn = QtWidgets.QPushButton("[ 🔊 VOICE: ON ]")
        self.tts_btn.setCheckable(True)
        self.tts_btn.setChecked(True)
        self.tts_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.tts_btn.setStyleSheet(
            "QPushButton { background-color: #0B101D; color: #475569; border: 1px solid #334155; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:checked { background-color: #071F12; color: #39FF14; border: 1px solid #39FF14; }"
        )
        self.tts_btn.toggled.connect(self._on_toggle_tts)
        sub_bar.addWidget(self.tts_btn)

        self.proactive_btn = QtWidgets.QPushButton("[ 📡 PROACTIVE ETA: ON ]")
        self.proactive_btn.setCheckable(True)
        self.proactive_btn.setChecked(True)
        self.proactive_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.proactive_btn.setStyleSheet(
            "QPushButton { background-color: #0B101D; color: #475569; border: 1px solid #334155; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:checked { background-color: #0B101D; color: #38BDF8; border: 1px solid #38BDF8; }"
        )
        self.proactive_btn.toggled.connect(self._on_toggle_proactive)
        sub_bar.addWidget(self.proactive_btn)

        self.narrator_btn = QtWidgets.QPushButton("[ 📖 NARRATOR: SILENT ]")
        self.narrator_btn.setCheckable(True)
        self.narrator_btn.setChecked(False)
        self.narrator_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.narrator_btn.setToolTip("When SILENT, the Chronicler writes your book in the background without speaking. Say 'Atlas, turn narrator on' to hear it.")
        self.narrator_btn.setStyleSheet(
            "QPushButton { background-color: #0B101D; color: #94A3B8; border: 1px solid #334155; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:checked { background-color: #1E1035; color: #E879F9; border: 1px solid #A855F7; }"
        )
        self.narrator_btn.toggled.connect(self._on_toggle_narrator_voice)
        sub_bar.addWidget(self.narrator_btn)

        self.mc_text_btn = QtWidgets.QPushButton("[ 💬 MC TEXT: OFF ]")
        self.mc_text_btn.setCheckable(True)
        self.mc_text_btn.setChecked(False)
        self.mc_text_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.mc_text_btn.setToolTip("Keep Minecraft chat completely clean (OFF) or mirror short callouts to in-game chat (ON)")
        self.mc_text_btn.setStyleSheet(
            "QPushButton { background-color: #0B101D; color: #94A3B8; border: 1px solid #334155; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:checked { background-color: #071F12; color: #39FF14; border: 1px solid #39FF14; }"
        )
        self.mc_text_btn.toggled.connect(self._on_toggle_mc_text)
        sub_bar.addWidget(self.mc_text_btn)

        sub_bar.addWidget(QtWidgets.QLabel("🎙️ PTT:"))
        self.ptt_combo = QtWidgets.QComboBox()
        self.ptt_combo.addItems(list(PTT_KEYS.keys()))
        self.ptt_combo.setStyleSheet("background-color: #0B101D; color: #FACC15; border: 1px solid #FACC15; padding: 3px 8px; font-weight: bold;")
        self.ptt_combo.currentTextChanged.connect(self._on_ptt_key_changed)
        sub_bar.addWidget(self.ptt_combo)

        self.pin_btn = QtWidgets.QPushButton("[ PIN ALWAYS-ON-TOP HUD ]")
        self.pin_btn.setCheckable(True)
        self.pin_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.pin_btn.setStyleSheet(
            "QPushButton { background-color: #0B101D; color: #00D2FF; border: 1px solid #00D2FF; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:checked { background-color: #00D2FF; color: #060913; }"
        )
        self.pin_btn.toggled.connect(self._toggle_always_on_top)
        sub_bar.addWidget(self.pin_btn)
        root.addLayout(sub_bar)

        mode_bar = QtWidgets.QHBoxLayout()
        mode_bar.setSpacing(8)
        mode_bar.addWidget(QtWidgets.QLabel("⚙️ ENGINE MODE:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["PRODUCTION", "STORY", "HYBRID"])
        self.mode_combo.setCurrentText(self.chronicler.mode)
        self.mode_combo.setStyleSheet(
            "background-color: #0B101D; color: #39FF14; border: 1px solid #39FF14; padding: 4px 10px; font-weight: bold;"
        )
        self.mode_combo.currentTextChanged.connect(self._on_mode_combo_changed)
        mode_bar.addWidget(self.mode_combo)

        mode_bar.addWidget(QtWidgets.QLabel("  📜 STORY THEME:"))
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(list(STORY_THEMES.keys()))
        self.theme_combo.setCurrentText(self.chronicler.theme_name)
        self.theme_combo.setMinimumWidth(260)
        self.theme_combo.setStyleSheet(
            "background-color: #0B101D; color: #E879F9; border: 1px solid #A855F7; padding: 4px 10px; font-weight: bold;"
        )
        self.theme_combo.currentTextChanged.connect(self._on_theme_combo_changed)
        mode_bar.addWidget(self.theme_combo)

        self.write_beat_btn = QtWidgets.QPushButton("[ ✍️ LOG CHAPTER BEAT ]")
        self.write_beat_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.write_beat_btn.setStyleSheet(
            "QPushButton { background-color: #1E1035; color: #E879F9; border: 1px solid #A855F7; padding: 4px 12px; font-weight: bold; } "
            "QPushButton:hover { background-color: #A855F7; color: #060913; }"
        )
        self.write_beat_btn.clicked.connect(lambda: self._trigger_chronicle_beat("Player Chronicle Entry"))
        mode_bar.addWidget(self.write_beat_btn)

        self.open_book_btn = QtWidgets.QPushButton("[ 📖 OPEN BOOK ]")
        self.open_book_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.open_book_btn.setStyleSheet(
            "QPushButton { background-color: #0B101D; color: #38BDF8; border: 1px solid #38BDF8; padding: 4px 12px; font-weight: bold; } "
            "QPushButton:hover { background-color: #38BDF8; color: #060913; }"
        )
        self.open_book_btn.clicked.connect(self._open_chronicle_book)
        mode_bar.addWidget(self.open_book_btn)

        self.task_hud_btn = QtWidgets.QPushButton('[ 📋 TASKS HUD (J) ]')
        self.task_hud_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.task_hud_btn.setToolTip('Open/Hide the Categorized In-Game Task & Dependency Overlay (Hotkey: J)')
        self.task_hud_btn.setStyleSheet(
            'QPushButton { background-color: #071F12; color: #39FF14; border: 1px solid #39FF14; padding: 4px 12px; font-weight: bold; } '
            'QPushButton:hover { background-color: #39FF14; color: #060913; }'
        )
        self.task_hud_btn.clicked.connect(self._toggle_task_overlay_hud)
        mode_bar.addWidget(self.task_hud_btn)

        mode_bar.addWidget(QtWidgets.QLabel("  🎯 ACTIVE TASK:"))
        self.task_edit = QtWidgets.QLineEdit("cassiterite, copper, tin, bronze, iron")
        self.task_edit.setPlaceholderText("Current modpack goal/ores (e.g. cassiterite, bronze steam, redstone)...")
        self.task_edit.setMinimumWidth(240)
        self.task_edit.setStyleSheet("background-color: #060913; color: #39FF14; border: 1px solid #39FF14; padding: 4px 8px; font-weight: bold;")
        self.task_edit.textChanged.connect(self._on_active_task_changed)
        mode_bar.addWidget(self.task_edit, 1)

        mode_bar.addStretch(1)
        self.token_hud_lbl = QtWidgets.QLabel("🪙 HQ SYNC: 0 tok (Voice: 0 | Keys/Chat: 0 | Story: 0)")
        self.token_hud_lbl.setStyleSheet("color: #FACC15; font-weight: bold; font-size: 9pt; padding-right: 12px;")
        mode_bar.addWidget(self.token_hud_lbl)

        self.book_status_lbl = QtWidgets.QLabel(f"📖 Ch.{self.chronicler.chapter_num} ({self.chronicler.entry_num} entries)")
        self.book_status_lbl.setStyleSheet("color: #E879F9; font-weight: bold; font-size: 9.5pt;")
        mode_bar.addWidget(self.book_status_lbl)
        root.addLayout(mode_bar)

        self.hazard_banner = QtWidgets.QLabel("✔ ALL BASE SYSTEMS NOMINAL — Zero Explosion / Voltage Hazards Detected")
        self.hazard_banner.setStyleSheet(
            "background-color: #071F12; color: #39FF14; border: 1px solid #15803D; "
            "border-radius: 4px; padding: 6px 10px; font-size: 9.5pt; font-weight: bold;"
        )
        root.addWidget(self.hazard_banner)

        split = QtWidgets.QHBoxLayout()
        split.setSpacing(10)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(8)

        tele_group = QtWidgets.QGroupBox("LIVE PRE-RENDER TELEMETRY & 3x3 ORE-VEIN RADAR")
        tele_group.setStyleSheet(
            "QGroupBox { background-color: #0B101D; border: 1px solid #1E293B; padding-top: 18px; font-weight: bold; color: #38BDF8; }"
        )
        tg_layout = QtWidgets.QVBoxLayout(tele_group)

        self.coords_lbl = QtWidgets.QLabel()
        self.coords_lbl.setStyleSheet("color: #F8FAFC; font-size: 9pt; font-weight: bold;")
        tg_layout.addWidget(self.coords_lbl)

        self.ore_radar_lbl = QtWidgets.QLabel()
        self.ore_radar_lbl.setWordWrap(True)
        self.ore_radar_lbl.setStyleSheet(
            "background-color: #060913; color: #39FF14; border: 1px solid #1E293B; padding: 8px; font-size: 9.5pt;"
        )
        tg_layout.addWidget(self.ore_radar_lbl)

        sim_row = QtWidgets.QHBoxLayout()
        self.walk_btn = QtWidgets.QPushButton("▶ SIMULATE WALK")
        self.walk_btn.setCheckable(True)
        self.walk_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.walk_btn.setStyleSheet(
            "QPushButton { background-color: #0F172A; color: #39FF14; border: 1px solid #39FF14; padding: 5px 10px; font-weight: bold; } "
            "QPushButton:checked { background-color: #39FF14; color: #060913; }"
        )
        self.walk_btn.toggled.connect(lambda on: self.sim_timer.start() if on else self.sim_timer.stop())
        sim_row.addWidget(self.walk_btn)

        self.hazard_sim_btn = QtWidgets.QPushButton("⚡ TEST BASE HAZARD")
        self.hazard_sim_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.hazard_sim_btn.setStyleSheet(
            "background-color: #0F172A; color: #FF003C; border: 1px solid #FF003C; padding: 5px 10px; font-weight: bold;"
        )
        self.hazard_sim_btn.clicked.connect(self._cycle_simulated_hazard)
        sim_row.addWidget(self.hazard_sim_btn)

        self.clear_haz_btn = QtWidgets.QPushButton("RESET SAFE")
        self.clear_haz_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.clear_haz_btn.setStyleSheet(
            "background-color: #0F172A; color: #CBD5E1; border: 1px solid #475569; padding: 5px 8px;"
        )
        self.clear_haz_btn.clicked.connect(self._reset_hazards)
        sim_row.addWidget(self.clear_haz_btn)

        tg_layout.addLayout(sim_row)
        left_col.addWidget(tele_group)

        vp_group = QtWidgets.QGroupBox("DISCOVERED VEINS & WAYPOINTS (512M SERVER RADAR + JOURNEYMAP)")
        vp_group.setStyleSheet(
            "QGroupBox { background-color: #0B101D; border: 1px solid #1E293B; padding-top: 18px; font-weight: bold; color: #A855F7; }"
        )
        vp_layout = QtWidgets.QVBoxLayout(vp_group)

        vp_filter_row = QtWidgets.QHBoxLayout()
        self.vp_filter_edit = QtWidgets.QLineEdit()
        self.vp_filter_edit.setPlaceholderText("Filter generated veins/waypoints (e.g. cassiterite, copper, iron, home)...")
        self.vp_filter_edit.setStyleSheet("background-color: #060913; color: #F8FAFC; border: 1px solid #334155; padding: 4px 8px;")
        self.vp_filter_edit.textChanged.connect(self._refresh_telemetry_views)
        vp_filter_row.addWidget(self.vp_filter_edit, 1)

        sync_cache_btn = QtWidgets.QPushButton("↻ SYNC CACHE")
        sync_cache_btn.setCursor(QtCore.Qt.PointingHandCursor)
        sync_cache_btn.setStyleSheet(
            "QPushButton { background-color: #0F172A; color: #A855F7; border: 1px solid #A855F7; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:hover { background-color: #A855F7; color: #060913; }"
        )
        sync_cache_btn.clicked.connect(self._force_rescan_caches)
        vp_filter_row.addWidget(sync_cache_btn)
        vp_layout.addLayout(vp_filter_row)

        self.vp_list_box = QtWidgets.QTextEdit()
        self.vp_list_box.setReadOnly(True)
        self.vp_list_box.setMinimumHeight(145)
        self.vp_list_box.setStyleSheet(
            "background-color: #060913; color: #E2E8F0; border: 1px solid #1E293B; padding: 6px; font-size: 9pt;"
        )
        vp_layout.addWidget(self.vp_list_box, 1)
        left_col.addWidget(vp_group, 1)

        oc_group = QtWidgets.QGroupBox("DETERMINISTIC GT OVERCLOCK & BATCH CALCULATOR")
        oc_group.setStyleSheet(
            "QGroupBox { background-color: #0B101D; border: 1px solid #1E293B; padding-top: 18px; font-weight: bold; color: #FFB000; }"
        )
        oc_layout = QtWidgets.QVBoxLayout(oc_group)

        form_row = QtWidgets.QHBoxLayout()
        self.eu_spin = QtWidgets.QSpinBox()
        self.eu_spin.setRange(1, 2000000)
        self.eu_spin.setValue(30)
        self.eu_spin.setPrefix("EU/t: ")
        self.eu_spin.setStyleSheet("background-color: #060913; color: #F8FAFC; border: 1px solid #334155; padding: 4px;")
        form_row.addWidget(self.eu_spin)

        self.ticks_spin = QtWidgets.QSpinBox()
        self.ticks_spin.setRange(1, 1000000)
        self.ticks_spin.setValue(200)
        self.ticks_spin.setPrefix("Ticks: ")
        self.ticks_spin.setStyleSheet("background-color: #060913; color: #F8FAFC; border: 1px solid #334155; padding: 4px;")
        form_row.addWidget(self.ticks_spin)

        self.tier_combo = QtWidgets.QComboBox()
        for t_name, t_eu in GT_VOLTAGE_TIERS:
            self.tier_combo.addItem(f"{t_name} ({t_eu} EU/t)", t_eu)
        self.tier_combo.setCurrentIndex(3)
        self.tier_combo.setStyleSheet("background-color: #060913; color: #38BDF8; border: 1px solid #334155; padding: 4px;")
        form_row.addWidget(self.tier_combo)

        self.perf_chk = QtWidgets.QCheckBox("Perfect OC (4x)")
        self.perf_chk.setStyleSheet("color: #CBD5E1;")
        form_row.addWidget(self.perf_chk)
        oc_layout.addLayout(form_row)

        self.oc_result_lbl = QtWidgets.QLabel()
        self.oc_result_lbl.setWordWrap(True)
        self.oc_result_lbl.setStyleSheet(
            "background-color: #060913; color: #E2E8F0; border: 1px solid #1E293B; padding: 8px; font-size: 9.5pt;"
        )
        oc_layout.addWidget(self.oc_result_lbl)

        self.eu_spin.valueChanged.connect(self._update_overclock_preview)
        self.ticks_spin.valueChanged.connect(self._update_overclock_preview)
        self.tier_combo.currentIndexChanged.connect(self._update_overclock_preview)
        self.perf_chk.toggled.connect(self._update_overclock_preview)
        self._update_overclock_preview()

        left_col.addWidget(oc_group)
        split.addLayout(left_col, 5)

        right_group = QtWidgets.QGroupBox("SOVEREIGN PIT CREW COMMS & ENGINE TELEMETRY")
        right_group.setStyleSheet(
            "QGroupBox { background-color: #0B101D; border: 1px solid #00D2FF; padding-top: 18px; font-weight: bold; color: #00D2FF; }"
        )
        rg_layout = QtWidgets.QVBoxLayout(right_group)

        crew_bar = QtWidgets.QHBoxLayout()
        crew_bar.addWidget(QtWidgets.QLabel("SPECIALIST:"))
        self.crew_combo = QtWidgets.QComboBox()
        self.crew_combo.addItems(list(CREW_ROSTER.keys()))
        self.crew_combo.setStyleSheet("background-color: #060913; color: #F8FAFC; border: 1px solid #00D2FF; padding: 4px 8px;")
        crew_bar.addWidget(self.crew_combo, 1)

        quick_ore_btn = QtWidgets.QPushButton("🧭 Vein Callout")
        quick_ore_btn.setCursor(QtCore.Qt.PointingHandCursor)
        quick_ore_btn.setStyleSheet("background-color: #0F172A; color: #38BDF8; border: 1px solid #0284C7; padding: 4px 8px;")
        quick_ore_btn.clicked.connect(
            lambda: self._dispatch_crew(
                "Geologist / Navigator",
                "Give me an immediate bearing and prospecting plan for the nearest ore vein center and summarize my closest discovered veins/waypoints.",
            )
        )
        crew_bar.addWidget(quick_ore_btn)

        clear_log_btn = QtWidgets.QPushButton("CLEAR")
        clear_log_btn.setCursor(QtCore.Qt.PointingHandCursor)
        clear_log_btn.setStyleSheet("background-color: #0F172A; color: #CBD5E1; border: 1px solid #334155; padding: 4px 8px;")
        clear_log_btn.clicked.connect(self.radio_console_clear)
        crew_bar.addWidget(clear_log_btn)
        rg_layout.addLayout(crew_bar)

        self.radio_console = QtWidgets.QTextEdit()
        self.radio_console.setReadOnly(True)
        self.radio_console.setFont(QtGui.QFont("Consolas", 9.5))
        self.radio_console.setStyleSheet(
            "background-color: #060913; color: #E2E8F0; border: 1px solid #1E293B; padding: 8px;"
        )
        rg_layout.addWidget(self.radio_console, 1)

        ask_row = QtWidgets.QHBoxLayout()
        self.ask_edit = QtWidgets.QLineEdit()
        self.ask_edit.setPlaceholderText("Ask Pit Crew (or hold ~ / Mouse4 in-game to talk, or press [  ]  \\)...")
        self.ask_edit.setStyleSheet("background-color: #060913; color: #F8FAFC; border: 1px solid #475569; padding: 6px;")
        self.ask_edit.returnPressed.connect(self._send_manual_query)
        ask_row.addWidget(self.ask_edit, 1)

        self.send_btn = QtWidgets.QPushButton("TRANSMIT")
        self.send_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.send_btn.setStyleSheet("background-color: #00D2FF; color: #060913; font-weight: bold; padding: 6px 14px; border: none;")
        self.send_btn.clicked.connect(self._send_manual_query)
        ask_row.addWidget(self.send_btn)
        rg_layout.addLayout(ask_row)

        split.addWidget(right_group, 7)
        root.addLayout(split, 1)

    def _test_active_specialist_voice(self):
        role = self.crew_combo.currentText()
        sample_lines = {
            "Geologist / Navigator": f"Geologist online. Locked onto chunk {self.telemetry.get('x', 0) >> 4}, {self.telemetry.get('z', 0) >> 4}. Ready for subsurface prospecting.",
            "Wayfinder / Map Specialist": f"Wayfinder standing by in {self.telemetry.get('biome', 'the frontier')}. Surface compass and waypoint vectors are locked.",
            "Chief Industrial Planner": "Industrial Planner online. Ready to calculate machine overclocks and batch processing routes.",
            "Weapon / Armor / Tool Specialist": f"Armorer here. Assessing your {self.telemetry.get('held_item', 'gear')} for progression-safe upgrades.",
            "Combat Specialist": f"Combat Coach on comms. Health is at {round(float(self.telemetry.get('health', 20.0))/2.0, 1)} hearts—watch your spacing.",
            "Safety & Voltage Officer": "Safety Officer monitoring base grid. Zero voltage or boiler hazards detected.",
            "Story Chronicler": f"The chronicler dipped his quill, recording a new chapter in {self.chronicler.theme_name}.",
        }
        line = sample_lines.get(role, "Atlas voice comms online.")
        self._append_engine_log("SYSTEM", f"🔊 Testing voice for [{role}]: \"{line}\"")
        if hasattr(self, "voice_engine"):
            self.voice_engine.speak(role, line)

    def _rescan_models(self):
        curr = self.model_combo.currentText()
        items = scan_training_models() or ["NO MODELS FOUND"]
        self.model_combo.clear()
        self.model_combo.addItems(items)
        if curr in items:
            self.model_combo.setCurrentText(curr)

    def _set_status_badge(self, state_code: str, text: str):
        color = "#39FF14" if state_code == "READY" else ("#FFB000" if state_code == "LOADING" else "#FF003C")
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(
            f"color: {color}; font-family: Consolas; font-size: 10pt; font-weight: bold; padding-right: 6px;"
        )

    def _on_engine_status_changed(self, port: str, state_code: str, message: str):
        if port == self.port_combo.currentText():
            self._set_status_badge(state_code, message)
            self.unload_btn.setEnabled(state_code in ("LOADING", "READY"))

    def _on_port_combo_changed(self, selected_port: str):
        self.llm_port = int(selected_port)
        state_code, summary = self.engine_mgr.get_port_summary(selected_port)
        self._set_status_badge(state_code, summary)
        self.unload_btn.setEnabled(state_code in ("LOADING", "READY"))
        self._poll_external_engine()

    def _poll_external_engine(self):
        if hasattr(self, "voice_engine") and hasattr(self, "f5_btn"):
            if self.voice_engine.is_f5_online():
                self.f5_btn.setText("● F5-TTS (:1237) ONLINE")
                self.f5_btn.setStyleSheet("background-color: #071F12; color: #39FF14; border: 1px solid #39FF14; padding: 4px 12px; font-weight: bold;")
            else:
                self.f5_btn.setText("[ 🧠 BOOT F5-TTS :1237 ]")
                self.f5_btn.setStyleSheet("background-color: #1E1035; color: #E879F9; border: 1px solid #A855F7; padding: 4px 12px; font-weight: bold;")

        port_str = str(self.llm_port)
        managed_state, managed_summary = self.engine_mgr.get_port_summary(port_str)
        if managed_state in ("LOADING", "READY"):
            self._set_status_badge(managed_state, managed_summary)
            self.unload_btn.setEnabled(True)
            return

        try:
            r = requests.get(f"http://{self.llm_host}:{self.llm_port}/v1/models", timeout=0.6)
            if r.status_code == 200:
                data = r.json().get("data", [])
                m_id = os.path.basename(data[0].get("id", "External Model")) if data else "External Model"
                short = m_id[:24] + ".." if len(m_id) > 26 else m_id
                self._set_status_badge("READY", f"ONLINE (:{self.llm_port}) | {short} [External]")
                self.unload_btn.setEnabled(True)
                return
        except Exception:
            pass

        self._set_status_badge("OFFLINE", f"PORT {self.llm_port}: OFFLINE")
        self.unload_btn.setEnabled(False)

    def _force_rescan_caches(self):
        self.cfg = load_config()
        self.cache_index.set_instance_dir(self.cfg.get("gtnh_instance_dir", ""))
        self.cache_index.refresh_if_needed(force=True)
        self._refresh_telemetry_views()
        self._append_engine_log(
            "SYSTEM",
            f"Synced caches: {len(self.cache_index._server_veins)} Server Vein(s), {len(self.cache_index.waypoints)} JourneyMap Waypoint(s).",
        )

    def radio_console_clear(self):
        self.radio_console.clear()
        self._mid_stream = False

    def _scroll_console(self):
        sb = self.radio_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    @QtCore.Slot(str, str)
    def _append_engine_log_slot(self, category: str, message: str):
        self._append_engine_log(category, message)

    def _append_engine_log(self, category: str, message: str):
        cursor = self.radio_console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        if self._mid_stream:
            cursor.insertText("\n")
            self._mid_stream = False

        color = self.LOG_COLORS.get(category, "#94A3B8")

        ts_fmt = QtGui.QTextCharFormat()
        ts_fmt.setForeground(QtGui.QColor("#64748B"))
        cursor.insertText(f"[{datetime.now().strftime('%H:%M:%S')}] ", ts_fmt)

        tag_fmt = QtGui.QTextCharFormat()
        tag_fmt.setForeground(QtGui.QColor(color))
        tag_fmt.setFontWeight(QtGui.QFont.Bold)
        cursor.insertText(f"[{category}] ", tag_fmt)

        msg_fmt = QtGui.QTextCharFormat()
        msg_fmt.setForeground(QtGui.QColor(color if category in ("READY", "WARN", "ERROR") else "#CBD5E1"))
        cursor.insertText(f"{message}\n", msg_fmt)
        self._scroll_console()

    def _toggle_always_on_top(self, checked: bool):
        flags = self.windowFlags()
        if checked:
            self.setWindowFlags(flags | QtCore.Qt.WindowStaysOnTopHint)
            self.setWindowOpacity(0.93)
        else:
            self.setWindowFlags(flags & ~QtCore.Qt.WindowStaysOnTopHint)
            self.setWindowOpacity(1.0)
        self.show()

    def _on_toggle_tts(self, checked: bool):
        self.tts_btn.setText("[ 🔊 VOICE: ON ]" if checked else "[ 🔇 VOICE: OFF ]")
        if hasattr(self, "voice_engine"):
            self.voice_engine.tts_enabled = checked

    def _on_toggle_proactive(self, checked: bool):
        self.proactive_btn.setText("[ 📡 PROACTIVE ETA: ON ]" if checked else "[ 📡 PROACTIVE ETA: OFF ]")
        if hasattr(self, "voice_engine"):
            self.voice_engine.proactive_enabled = checked

    def _on_ptt_key_changed(self, mode_text: str):
        if hasattr(self, "voice_engine"):
            self.voice_engine.ptt_mode = mode_text

    def _on_mode_combo_changed(self, mode_str: str):
        self.chronicler.set_mode(mode_str)
        if hasattr(self, 'voice_engine'):
            self.voice_engine.engine_mode = mode_str
        colors = {"PRODUCTION": "#39FF14", "STORY": "#E879F9", "HYBRID": "#38BDF8"}
        col = colors.get(mode_str, "#39FF14")
        self.mode_combo.setStyleSheet(
            f"background-color: #0B101D; color: {col}; border: 1px solid {col}; padding: 4px 10px; font-weight: bold;"
        )
        if mode_str == "STORY":
            self.crew_combo.setCurrentText("Story Chronicler")
        banners = {
            "PRODUCTION": "§a[PRODUCTION MODE] §7DeepSeek Critical-Path Speedrun & Engineering Active.",
            "STORY":      f"§d[STORY MODE] §7Theme: §f{self.chronicler.theme_name} §7(Live Book Chronicler Active).",
            "HYBRID":     f"§b[HYBRID MODE] §7Tactical GTNH Progression + §f{self.chronicler.theme_name} §7Active.",
        }
        self._append_engine_log("SYSTEM", f"Engine Mode -> {mode_str} | Theme: {self.chronicler.theme_name}")
        self.tcp_server.send_ingame_chat(banners.get(mode_str, ""))

    def _on_theme_combo_changed(self, theme_str: str):
        self.chronicler.set_theme(theme_str)
        self.book_status_lbl.setText(f"📖 Ch.{self.chronicler.chapter_num} ({self.chronicler.entry_num} entries)")
        self._append_engine_log("SYSTEM", f"Story Theme -> {theme_str} ({os.path.basename(self.chronicler.get_book_filepath())})")
        self.tcp_server.send_ingame_chat(f"§d[ATLAS CHRONICLER] §7Active Book Theme: §f{theme_str}")

    def _trigger_chronicle_beat(self, reason: str = "Chronicle Milestone"):
        self._pending_book_reason = reason
        prompt = (
            f"Write the next vivid narrative passage for our ongoing book (Milestone: {reason}). "
            f"Describe my current surroundings in {self.telemetry.get('biome', 'the frontier')} at ({self.telemetry.get('x')}, {self.telemetry.get('y')}, {self.telemetry.get('z')}), "
            f"what I am holding ({self.telemetry.get('held_item')}), what I am looking at ({self.telemetry.get('crosshair_target')}), and how this moment advances our '{self.chronicler.theme_name}' journey."
        )
        role = "Story Chronicler" if self.chronicler.mode == "STORY" else self.crew_combo.currentText()
        self._dispatch_crew(role, prompt, trigger_source="Story Chronicle")

    def _open_chronicle_book(self):
        fp = self.chronicler.get_book_filepath()
        if os.path.isfile(fp) and os.name == "nt":
            os.startfile(fp)
        else:
            self._append_engine_log("WARN", f"Book file not found yet: {fp}")

    def _cycle_engine_mode(self):
        order = ["PRODUCTION", "STORY", "HYBRID"]
        cur = self.mode_combo.currentText()
        nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else "PRODUCTION"
        self.mode_combo.setCurrentText(nxt)

    def _route_role_from_query(self, q_low: str) -> str:
        if hasattr(self, "chronicler") and self.chronicler.mode == "PRODUCTION":
            act = self.task_mgr.get_active_task() if hasattr(self, "task_mgr") else None
            if act and "Mining" in act.get("category", "") and not any(k in q_low for k in ("circuit", "machine", "overclock", "explode", "boiler")):
                return "Geologist / Navigator"
        if any(k in q_low for k in ("waypoint", "journeymap", "map", "biome", "route", "path", "wayfinder", "home", "base", "lost", "portal", "village", "temple", "structure", "humidity", "climate", "crop", "direction", "far", "eta")):
            return "Wayfinder / Map Specialist"
        if any(k in q_low for k in ("pick", "pickaxe", "hammer", "tinker", "tool", "sword", "cleaver", "crossbow", "rapier", "armor", "shield", "backpack", "mining level", "modifier", "alumite", "bronze", "flint", "wrought")):
            return "Weapon / Armor / Tool Specialist"
        if any(k in q_low for k in ("mob", "creeper", "zombie", "skeleton", "spider", "infernal", "fight", "combat", "attack", "damage", "health", "heart", "heal", "kite", "survive", "blood moon", "darkness")):
            return "Combat Specialist"
        if any(k in q_low for k in ("rain", "boiler", "explode", "voltage", "wire", "cable", "amp", "hazard", "safe")):
            return "Safety & Voltage Officer"
        if any(k in q_low for k in ("vein", "ore", "chunk", "where", "find", "cassiterite", "chalcopyrite", "copper", "tin", "iron", "gold", "diamond", "redstone", "mica", "bauxite", "galena", "nickel", "coal", "quartz", "salt", "apatite")):
            return "Geologist / Navigator"
        return self.crew_combo.currentText()

    def _on_toggle_narrator_voice(self, checked: bool):
        self.narrator_btn.setText("[ 📖 NARRATOR: VOICE ON ]" if checked else "[ 📖 NARRATOR: SILENT ]")
        if hasattr(self, "voice_engine"):
            self.voice_engine.narrator_voice_enabled = checked
        state_str = "ENABLED (Narrator will read story passages aloud)" if checked else "SILENT (Book logs quietly in the background)"
        self._append_engine_log("SYSTEM", f"📖 Story Narrator Voice -> {state_str}")

    def _on_toggle_mc_text(self, checked: bool):
        self.mc_text_btn.setText("[ 💬 MC TEXT: ON ]" if checked else "[ 💬 MC TEXT: OFF ]")
        if hasattr(self, "voice_engine"):
            self.voice_engine.ingame_text_enabled = checked
        self._append_engine_log("SYSTEM", f"💬 In-Game Minecraft Chat Text -> {'ON' if checked else 'OFF (Clean Screen)'}")

    def _on_active_task_changed(self, text: str):
        if hasattr(self, "voice_engine"):
            self.voice_engine.active_task_filter = text.strip()

    def _toggle_task_overlay_hud(self):
        if not hasattr(self, "task_overlay"):
            return
        if self.task_overlay.isVisible():
            self.task_overlay.hide()
        else:
            self.task_overlay.refresh_tree()
            self.task_overlay.show()
            self.task_overlay.raise_()

    def _sync_active_task_to_engine(self):
        if not hasattr(self, "task_mgr"):
            return
        act = self.task_mgr.get_active_task()
        kw_filter = self.task_mgr.get_active_filter_keywords()
        if hasattr(self, "task_edit"):
            self.task_edit.blockSignals(True)
            self.task_edit.setText(act["title"] if act else kw_filter)
            self.task_edit.blockSignals(False)
        if hasattr(self, "voice_engine"):
            self.voice_engine.active_task_filter = kw_filter
            if act:
                self.voice_engine.active_task_title = act["title"]
                self.voice_engine.active_task_category = act["category"]
                self.voice_engine.active_lead_specialist = act["lead_specialist"]
                if self.chronicler.mode == "PRODUCTION":
                    self.crew_combo.setCurrentText(act["lead_specialist"])
        if hasattr(self, "task_overlay") and self.task_overlay.isVisible():
            self.task_overlay.refresh_tree()

    def _on_user_added_task(self, raw_task_text: str):
        task_obj = self.task_mgr.add_task(raw_task_text, auto_expand=True)
        self._sync_active_task_to_engine()
        lead = task_obj["lead_specialist"]
        sub_count = len(task_obj.get("subtasks", []))
        self._append_engine_log(
            "READY",
            f"📋 Added Task [{task_obj['category']}]: '{task_obj['title']}' ({sub_count} sub-steps) | Lead: {lead}",
        )
        # Tactical 1-sentence confirmation from the Lead Specialist
        if hasattr(self, "voice_engine"):
            self.voice_engine.speak(lead, f"Logged {task_obj['title']} under {lead.split('/')[0].strip()}.")
        # Behind-the-scenes: Storyteller weaves the requisition into the book!
        sub_texts = [s["text"] for s in task_obj.get("subtasks", [])]
        story_title, story_prompt = self.chronicler.build_task_requisition_story_prompt(
            task_obj["title"], sub_texts, self.telemetry
        )
        self._spawn_background_story_beat(story_title, story_prompt)

    def _on_request_ai_task_breakdown(self, goal_text: str):
        task_obj = self.task_mgr.add_task(goal_text, auto_expand=True)
        self._sync_active_task_to_engine()
        prompt = (
            f"Break down the GTNH 2.9.0 goal '{goal_text}' into the 3 most critical, progression-safe steps "
            f"for my current stage and tell me what to mine or craft first in 2 short sentences."
        )
        self._dispatch_crew(task_obj["lead_specialist"], prompt, trigger_source="Cockpit UI")

    def _spawn_background_story_beat(self, story_title: str, story_prompt: str):
        if self.chronicler.mode == "PRODUCTION":
            return
        if self._bg_story_worker and self._bg_story_worker.isRunning():
            return
        sys_prompt, temp = self.chronicler.build_system_prompt("", is_background_book=True)
        self._bg_story_worker = BackgroundStoryWorker(
            self.llm_host,
            self.llm_port,
            sys_prompt,
            temp,
            dict(self.telemetry),
            story_title,
            story_prompt,
            self,
        )
        self._bg_story_worker.storyBeatReady.connect(self._on_background_story_ready)
        self._bg_story_worker.start()

    def _on_background_story_ready(self, story_title: str, prose: str, prompt_toks: int, comp_toks: int):
        book_fp = self.chronicler.append_book_entry(story_title, prose, self.telemetry)
        self.book_status_lbl.setText(f"📖 Ch.{self.chronicler.chapter_num} ({self.chronicler.entry_num} entries)")
        self.hq_logger.log_llm_call(
            trigger_source="Story Chronicle",
            engine_mode=self.chronicler.mode,
            specialist="Story Chronicler",
            model_name=self.model_combo.currentText(),
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
        )
        self._refresh_token_hud_badge()
        self._append_engine_log(
            "SYSTEM",
            f"📖 [BACKGROUND CHRONICLER -> {os.path.basename(book_fp)}] ({story_title}): {prose}",
        )
        if hasattr(self, "voice_engine") and self.voice_engine.narrator_voice_enabled:
            self.voice_engine.speak("Story Chronicler", prose, force_narrator=True)

    def _handle_detected_behavior_event(self, ev: dict):
        kind = ev.get("kind", "")
        title = ev.get("title", "World Event")
        story_prompt = ev.get("story_prompt", "")
        prod_callout = ev.get("prod_callout", "")

        self._append_engine_log("SYSTEM", f"👁️ Detected Player Behavior [{kind}]: {title}")

        # In PRODUCTION mode, call out inefficiencies (spinning in circles / respawn) via the active lead specialist!
        if self.chronicler.mode == "PRODUCTION" and prod_callout:
            lead = getattr(self.voice_engine, "active_lead_specialist", "Geologist / Navigator")
            self._append_engine_log("WARN", f"[PRODUCTION AUDITOR -> {lead}]: {prod_callout}")
            if hasattr(self, "voice_engine"):
                self.voice_engine.speak(lead, prod_callout)

        # Always weave the behind-the-scenes story beat into the Markdown book!
        if story_prompt:
            self._spawn_background_story_beat(title, story_prompt)

    def _handle_meta_voice_commands(self, text: str) -> bool:
        """Intercepts meta voice/chat commands to toggle the narrator or update the active task."""
        low = text.lower().strip()
        if any(p in low for p in ("narrator on", "turn on narrator", "turn the narrator on", "enable narrator", "start narrating")):
            self.narrator_btn.setChecked(True)
            if hasattr(self, "voice_engine"):
                self.voice_engine.speak("Story Chronicler", "The Chronicler is awake and listening, ready to voice your journey.", force_narrator=True)
            return True
        if any(p in low for p in ("narrator off", "turn off narrator", "turn the narrator off", "disable narrator", "mute narrator", "silence narrator", "stop narrating")):
            self.narrator_btn.setChecked(False)
            if hasattr(self, "voice_engine"):
                self.voice_engine.speak("Geologist / Navigator", "Narrator muted. Continuing silent background logging.")
            return True
        if any(p in low for p in ("read the story", "read last chapter", "read the book", "what happened in the story")):
            if self.chronicler.story_so_far:
                latest = self.chronicler.story_so_far[-1].split("]: ", 1)[-1]
                self._append_engine_log("READY", f"📖 Reading latest chronicle entry: {latest}")
                if hasattr(self, "voice_engine"):
                    self.voice_engine.speak("Story Chronicler", latest, force_narrator=True)
            else:
                self._trigger_chronicle_beat("Voice Requested Narration")
                if hasattr(self, "voice_engine"):
                    self.voice_engine.narrator_voice_enabled = True
                    self.narrator_btn.setChecked(True)
            return True
        if any(p in low for p in ("show tasks", "open tasks", "open task list", "show task list", "hide tasks", "close tasks")):
            self._toggle_task_overlay_hud()
            return True
        m_add = re.search(
            r"(?:add\s+(.+?)\s+to\s+(?:the\s+|my\s+)?(?:task\s+)?list|add\s+task\s+(.+)|set\s+task\s+to\s+(.+)|our\s+task\s+is\s+(.+)|need\s+to\s+mine\s+(.+))",
            low,
        )
        if m_add:
            raw_goal = next((g for g in m_add.groups() if g), "").strip(" .!?")
            if raw_goal:
                if "need to mine" in low and not raw_goal.startswith("mine"):
                    raw_goal = f"Mine {raw_goal}"
                self._on_user_added_task(raw_goal)
                return True
        return False

    def _on_voice_transcript(self, transcript: str):
        if self._handle_meta_voice_commands(transcript):
            return
        q_low = transcript.lower()
        role = self._route_role_from_query(q_low)
        self._dispatch_crew(role, transcript, trigger_source="Voice PTT")

    def _on_external_telemetry(self, packet: dict):
        action = packet.pop("client_action", None)
        query = packet.pop("client_query", "")
        srv = packet.get("server_factory") or {}
        if srv.get("generated_veins"):
            self.cache_index.ingest_server_veins(srv["generated_veins"])
        self.telemetry.update(packet)
        bx, bz = int(self.telemetry.get("x", 0)), int(self.telemetry.get("z", 0))
        self.telemetry["discovered_veins_summary"] = self.cache_index.query_near_player(
            bx, bz, self.telemetry.get("dimension", 0), max_veins=15, max_waypoints=8
        )
        self._refresh_telemetry_views()

        if action == "HOTKEY_VEIN_CALLOUT":
            self._dispatch_crew("Geologist / Navigator", query, trigger_source="In-Game Hotkey")
        elif action == "HOTKEY_INSPECT_TARGET":
            self._dispatch_crew("Chief Industrial Planner", query, trigger_source="In-Game Hotkey")
        elif action == "HOTKEY_CYCLE_MODE":
            self._cycle_engine_mode()
        elif action == "CHAT_QUERY" and query:
            if self._handle_meta_voice_commands(query):
                return
            if query.lower().startswith("chronicle") or query.lower().startswith("story"):
                note = query.split(" ", 1)[1].strip() if " " in query else "Player Journal Entry"
                self._trigger_chronicle_beat(note)
                return
            q_low = query.lower()
            role = self._route_role_from_query(q_low)
            self._dispatch_crew(role, query, trigger_source="In-Game /atlas")

    def _tick_walk_sim(self):
        self._sim_step += 1
        self.telemetry["x"] += 3
        self.telemetry["z"] += 2 if (self._sim_step % 3) else -1
        self.telemetry["player_speed_bps"] = 4.3
        self._refresh_telemetry_views()

    def _cycle_simulated_hazard(self):
        self._hazard_idx = (self._hazard_idx % 3) + 1
        self._reset_hazards(refresh=False)
        if self._hazard_idx == 1:
            self.telemetry["is_raining"] = True
            self.telemetry["exposed_gt_machines"] = 2
        elif self._hazard_idx == 2:
            self.telemetry["crosshair_machine_max_eu_t"] = 128
            self.telemetry["crosshair_cable_eu_t"] = 512
        else:
            self.telemetry["boiler_temp_c"] = 485.0
            self.telemetry["boiler_water_mb"] = 0
        self._refresh_telemetry_views()
        self._dispatch_crew(
            "Safety & Voltage Officer",
            "CRITICAL ALARM TRIGGERED! Diagnose the active hazard in my telemetry and tell me the exact sequence to save my base!",
            trigger_source="Proactive / Hazard",
        )

    def _reset_hazards(self, refresh=True):
        self.telemetry["is_raining"] = False
        self.telemetry["is_thundering"] = False
        self.telemetry["exposed_gt_machines"] = 0
        self.telemetry["crosshair_machine_max_eu_t"] = 32
        self.telemetry["crosshair_cable_eu_t"] = 32
        self.telemetry["boiler_temp_c"] = 85.0
        self.telemetry["boiler_water_mb"] = 4000
        if refresh:
            self._refresh_telemetry_views()

    def _refresh_telemetry_views(self):
        bx, by, bz = int(self.telemetry["x"]), int(self.telemetry["y"]), int(self.telemetry["z"])
        ore = resolve_nearest_ore_center(bx, bz)
        hazards = evaluate_safety_hazards(self.telemetry)

        if hasattr(self, "voice_engine"):
            self.voice_engine.evaluate_proactive_tripwires(self.telemetry, ore, hazards)
        if hasattr(self, "chronicler"):
            ev = self.chronicler.detect_live_player_behavior(self.telemetry, ore)
            if ev:
                self._handle_detected_behavior_event(ev)

        srv = self.telemetry.get("server_factory") or {}
        tps_str = (
            f"TPS: {srv.get('server_tps', '--')} ({srv.get('server_mspt', '--')}ms)  |  "
            f"GT Machines: {srv.get('gt_machines_active', 0)}/{srv.get('gt_machines_loaded', 0)}  |  "
            f"Seed: {srv.get('world_seed', 'Local')}"
        )
        spd = float(self.telemetry.get("player_speed_bps", 0.0))
        hp = float(self.telemetry.get("health", 20.0))
        self.coords_lbl.setText(
            f"POS: X={bx}  Y={by}  Z={bz}  |  CHUNK: ({ore['current_chunk_x']}, {ore['current_chunk_z']})  |  {self.telemetry.get('dimension')}  |  BIOME: {self.telemetry.get('biome', 'Unknown')}\n"
            f"TARGET: {self.telemetry.get('crosshair_target', 'Air')}  |  HELD: {self.telemetry.get('held_item', 'Empty')}  |  SPD: {spd:.1f} m/s  |  HP: {hp/2.0:.1f}❤  |  {tps_str}"
        )

        if ore["is_in_ore_chunk"]:
            zone_tag = "🎯 INSIDE VALID 3x3 ORE-VEIN CENTER CHUNK!"
            radar_color = "#39FF14"
        else:
            zone_tag = "SCANNING — Outside Ore Center Chunk"
            radar_color = "#38BDF8"

        self.ore_radar_lbl.setStyleSheet(
            f"background-color: #060913; color: {radar_color}; border: 1px solid #1E293B; padding: 8px; font-size: 9.5pt;"
        )
        self.ore_radar_lbl.setText(
            f"{zone_tag}\n"
            f"Nearest Vein Center : Chunk ({ore['target_chunk_x']}, {ore['target_chunk_z']}) ➔ Block (X={ore['target_block_x']}, Z={ore['target_block_z']})\n"
            f"Vector & Distance   : {ore['distance_blocks']} blocks  |  Heading: {ore['compass']}  (ΔX={ore['delta_x']:+d}, ΔZ={ore['delta_z']:+d})"
        )

        filt = self.vp_filter_edit.text() if hasattr(self, "vp_filter_edit") else ""
        snap = self.cache_index.query_near_player(
            bx, bz, self.telemetry.get("dimension", 0), search_filter=filt, max_veins=8, max_waypoints=6
        )
        lines = [
            f"Cached Total: {snap['total_veins_cached']} Server Vein(s) | {snap['total_waypoints_cached']} JourneyMap Waypoint(s)  [Dim {snap['dimension_id']}]"
        ]
        if snap["nearest_veins"]:
            lines.append("\n--- NEAREST GENERATED ORE VEINS (512M SERVER RADAR) ---")
            for v in snap["nearest_veins"]:
                lines.append(
                    f"• {v['name']} @ (X={v['x']}, Z={v['z']}) | {v['distance_blocks']}m {v['compass']} | Y:{v['y_range']} ({v['ores']})"
                )
        else:
            lines.append("\n• Waiting for ServerCache vein sync (ensure updated aadynamicstelemetry-1.0.0.jar is on BisectHosting)...")

        if snap["nearest_waypoints"]:
            lines.append("\n--- NEAREST JOURNEYMAP WAYPOINTS ---")
            for w in snap["nearest_waypoints"]:
                lines.append(
                    f"• [{w['category']}] {w['name']} @ (X={w['x']}, Y={w['y']}, Z={w['z']}) | {w['distance_blocks']}m {w['compass']}"
                )

        if hasattr(self, "vp_list_box"):
            self.vp_list_box.setPlainText("\n".join(lines))

        if hazards:
            msgs = "  ||  ".join(h["message"] for h in hazards)
            self.hazard_banner.setText(f"🚨 {msgs}")
            self.hazard_banner.setStyleSheet(
                "background-color: #2A0810; color: #FF003C; border: 1px solid #FF003C; "
                "border-radius: 4px; padding: 6px 10px; font-size: 9.5pt; font-weight: bold;"
            )
        else:
            self.hazard_banner.setText("✔ ALL BASE SYSTEMS NOMINAL — Zero Explosion / Voltage Hazards Detected")
            self.hazard_banner.setStyleSheet(
                "background-color: #071F12; color: #39FF14; border: 1px solid #15803D; "
                "border-radius: 4px; padding: 6px 10px; font-size: 9.5pt; font-weight: bold;"
            )

    def _update_overclock_preview(self):
        eu = self.eu_spin.value()
        ticks = self.ticks_spin.value()
        tier_eu = int(self.tier_combo.currentData())
        perf = self.perf_chk.isChecked()
        try:
            res = calculate_gt_overclock(eu, ticks, tier_eu, perf)
            self.oc_result_lbl.setText(
                f"Min Tier: {res['recipe_min_tier']}  ➔  Machine: {res['machine_tier']}  |  Overclocks: {res['overclock_count']}x\n"
                f"Final Draw: {res['final_eu_t']:,} EU/t  |  Duration: {res['final_ticks']} ticks ({res['duration_sec']}s)  |  Total Energy: {res['total_eu']:,} EU"
            )
        except ValueError as err:
            self.oc_result_lbl.setText(f"⚠️ {err}")

    def _send_manual_query(self):
        q = self.ask_edit.text().strip()
        if not q:
            return
        self.ask_edit.clear()
        if self._handle_meta_voice_commands(q):
            return
        role = self._route_role_from_query(q.lower())
        self._dispatch_crew(role, q)

    def _dispatch_crew(self, role_name: str, query: str, trigger_source: str = "Cockpit UI"):
        if self._active_worker and self._active_worker.isRunning():
            return
        if self.chronicler.mode == "STORY":
            role_name = "Story Chronicler"
        self.crew_combo.setCurrentText(role_name)

        bx, bz = int(self.telemetry["x"]), int(self.telemetry["z"])
        ore = resolve_nearest_ore_center(bx, bz)
        hazards = evaluate_safety_hazards(self.telemetry)

        ui_filter = self.vp_filter_edit.text().strip() if hasattr(self, "vp_filter_edit") else ""
        prospect_snap = self.cache_index.query_near_player(
            bx, bz, self.telemetry.get("dimension", 0), search_filter=ui_filter, max_veins=12, max_waypoints=8
        )

        q_low = query.lower()
        for kw in (
            "cassiterite", "chalcopyrite", "tetrahedrite", "magnetite", "copper", "tin", "iron",
            "coal", "redstone", "lapis", "diamond", "quartz", "salt", "bauxite", "galena", "nickel", "mica", "gold",
        ):
            if kw in q_low and kw != ui_filter.lower():
                kw_matches = self.cache_index.query_near_player(
                    bx, bz, self.telemetry.get("dimension", 0), search_filter=kw, max_veins=6, max_waypoints=6
                )
                prospect_snap[f"keyword_matches_{kw}"] = kw_matches["nearest_veins"]

        base_prompt = CREW_ROSTER.get(role_name, CREW_ROSTER["Geologist / Navigator"])
        self.telemetry['active_modpack_task'] = self.task_edit.text().strip() if hasattr(self, 'task_edit') else ''
        sys_prompt, temp = self.chronicler.build_system_prompt(base_prompt)
        self._last_dispatched_query = query
        self._active_trigger_source = trigger_source
        self._active_dispatched_role = role_name

        worker = CrewStreamWorker(
            self.llm_host,
            self.llm_port,
            role_name,
            dict(self.telemetry),
            ore,
            prospect_snap,
            hazards,
            query,
            custom_sys_prompt=sys_prompt,
            custom_temp=temp,
            parent=self,
        )
        self._active_worker = worker
        self.send_btn.setEnabled(False)

        worker.headerReady.connect(self._append_crew_header)
        worker.tokenDelta.connect(self._append_crew_delta)
        worker.usageStats.connect(self._on_worker_usage_stats)
        worker.streamDone.connect(self._on_crew_done)
        worker.streamError.connect(self._on_crew_error)
        worker.start()

    def _append_crew_header(self, role: str, query: str):
        cursor = self.radio_console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        if self._mid_stream:
            cursor.insertText("\n")
            self._mid_stream = False

        ts_fmt = QtGui.QTextCharFormat()
        ts_fmt.setForeground(QtGui.QColor("#64748B"))
        cursor.insertText(f"\n[{datetime.now().strftime('%H:%M:%S')}] ", ts_fmt)

        you_fmt = QtGui.QTextCharFormat()
        you_fmt.setForeground(QtGui.QColor("#FFB000"))
        you_fmt.setFontWeight(QtGui.QFont.Bold)
        cursor.insertText(f"YOU ➔ {role}: ", you_fmt)

        q_fmt = QtGui.QTextCharFormat()
        q_fmt.setForeground(QtGui.QColor("#CBD5E1"))
        cursor.insertText(f"{query}\n", q_fmt)

        role_fmt = QtGui.QTextCharFormat()
        role_fmt.setForeground(QtGui.QColor("#00D2FF"))
        role_fmt.setFontWeight(QtGui.QFont.Bold)
        cursor.insertText(f"[{role.upper()}]: ", role_fmt)
        self._scroll_console()

    def _append_crew_delta(self, text: str):
        cursor = self.radio_console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor("#F8FAFC"))
        cursor.insertText(text, fmt)
        self._mid_stream = not text.endswith("\n")
        self._scroll_console()

        self._ingame_line_buf += text
        self._tts_reply_buf += text
        while "\n" in self._ingame_line_buf:
            line, self._ingame_line_buf = self._ingame_line_buf.split("\n", 1)
            clean = line.strip().replace("**", "")
            if clean and hasattr(self, "voice_engine") and self.voice_engine.ingame_text_enabled:
                self.tcp_server.send_ingame_chat(f"§b[ATLAS] §f{clean}")

    def _on_crew_done(self):
        if self._ingame_line_buf.strip():
            clean = self._ingame_line_buf.strip().replace("**", "")
            if hasattr(self, "voice_engine") and self.voice_engine.ingame_text_enabled:
                self.tcp_server.send_ingame_chat(f"§b[ATLAS] §f{clean}")
        self._ingame_line_buf = ""

        full_spoken = self._tts_reply_buf.strip()
        self._tts_reply_buf = ""
        if full_spoken and hasattr(self, "voice_engine"):
            voice_role = "Story Chronicler" if self.chronicler.mode == "STORY" else self.crew_combo.currentText()
            self.voice_engine.speak(voice_role, full_spoken)

        if full_spoken and self.chronicler.mode in ("STORY", "HYBRID"):
            reason = self._pending_book_reason or self._last_dispatched_query or "Journey Entry"
            self._pending_book_reason = None
            book_fp = self.chronicler.append_book_entry(reason, full_spoken, self.telemetry)
            self.book_status_lbl.setText(f"📖 Ch.{self.chronicler.chapter_num} ({self.chronicler.entry_num} entries)")
            self._append_engine_log("SYSTEM", f"📖 Silently logged Ch.{self.chronicler.chapter_num} Entry {self.chronicler.entry_num} to {os.path.basename(book_fp)}")

        self._append_crew_delta("\n")
        self._mid_stream = False
        self.send_btn.setEnabled(True)

    def _on_crew_error(self, err_msg: str):
        self._append_engine_log("ERROR", err_msg)
        self.send_btn.setEnabled(True)

    def _on_audio_token_audit(self, audio_kind: str, role_or_source: str, text_content: str):
        self.hq_logger.log_audio_event(audio_kind, role_or_source, text_content)
        self.audioTokensLogged.emit()

    def _on_worker_usage_stats(self, prompt_tokens: int, completion_tokens: int):
        src = getattr(self, "_active_trigger_source", "Cockpit UI")
        role = getattr(self, "_active_dispatched_role", self.crew_combo.currentText())
        if self.chronicler.mode == "STORY" and src == "Cockpit UI":
            src = "Story Chronicle"
        self.hq_logger.log_llm_call(
            trigger_source=src,
            engine_mode=self.chronicler.mode,
            specialist=role,
            model_name=self.model_combo.currentText(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._refresh_token_hud_badge()
        self._append_engine_log(
            "SYSTEM",
            f"🪙 HQ Telemetry Synced [{src} | {self.chronicler.mode}]: {prompt_tokens + completion_tokens:,} tokens ({prompt_tokens:,} prompt + {completion_tokens:,} completion)",
        )

    def _refresh_token_hud_badge(self):
        t = self.hq_logger.session_totals
        voice_t = t.get("Voice PTT", 0) + t.get("Audio (STT/TTS)", 0)
        ingame_t = t.get("In-Game Hotkey", 0) + t.get("In-Game /atlas", 0) + t.get("Proactive / Hazard", 0)
        story_t = t.get("Story Chronicle", 0)
        self.token_hud_lbl.setText(
            f"🪙 HQ SYNC: {t['total']:,} tok (🎙️Voice/TTS: {voice_t:,} | 🎮Keys/Chat: {ingame_t:,} | 📖Story: {story_t:,})"
        )

    def closeEvent(self, event):
        self.tcp_server.stop()
        self.tcp_server.wait(1000)
        if hasattr(self, "voice_engine"):
            self.voice_engine.stop_f5_server()
        self.engine_mgr.unload_all()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = PitCrewWindow()
    win.show()
    sys.exit(app.exec())