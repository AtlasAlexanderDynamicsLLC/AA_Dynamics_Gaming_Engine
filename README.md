# AA Dynamics — AI Cognitive Engine (`v0.1.0-alpha`)

Stateful, multi-agent AI cognitive overlay and zero-shot neural voice telemetry engine developed by **Atlas & Alexander Dynamics LLC**.

## Core Architecture (`v0.1.0-alpha` Baseline)
* **Cognitive Bridge & UI (`pit_crew_bridge.py`):** Tri-mode execution router (`PRODUCTION`, `STORY`, `HYBRID`), live TCP telemetry listener (`:8765`), and standalone `llama-server.exe` controller (`:1234`).
* **Neural Audio Cortex (`tts_server.py` & `core/voice_comms.py`):** Local F5-TTS (`:1237`) zero-shot voice synthesis paired with CPU `faster-whisper` Push-to-Talk speech recognition.
* **Behavioral Story Chronicler (`core/story_chronicler.py`):** Real-time kinematic and event novelist (active in `STORY` and `HYBRID` modes; hard-disabled in `PRODUCTION` mode).
* **Task & Dependency HUD (`core/task_manager.py`):** Categorized in-game task overlay (`J` key) with automatic prerequisite expansion.
* **HQ Telemetry Sync (`core/hq_token_logger.py`):** Real-time SQLite WAL token and cost-avoidance logging into AA Dynamics HQ (`atlas_telemetry.db`).