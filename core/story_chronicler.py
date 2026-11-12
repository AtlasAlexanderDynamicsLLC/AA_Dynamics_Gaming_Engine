import os
import re
import math
import json
import time
from collections import deque
from datetime import datetime
from typing import Any

ENGINE_ROOT = r"E:\SovereignEngine\AA_Dynamics_Gaming_Engine"
CHRONICLES_DIR = os.path.join(ENGINE_ROOT, "chronicles")
STATE_FILE = os.path.join(CHRONICLES_DIR, "story_state.json")

STORY_THEMES = {
    "Solo Village Builder & Homestead": {
        "slug": "Village_Homestead",
        "book_title": "The Architect of the Quiet Frontier",
        "story_prompt": (
            "You are the Royal Chronicler & Master Narrator writing 'The Architect of the Quiet Frontier'. "
            "The player is a solitary master builder and homesteader carving a thriving village and sanctuary out of the wilderness. "
            "CRITICAL RULE: Never break the fourth wall and never mention 'the player', 'tasks', or 'video games'. "
            "Translate every real-time behavior—spinning in circles, requisitioning 500 ore, fighting monsters, or dying and waking at the hearth—"
            "into vivid, grounded novel prose (2-4 evocative sentences) with an in-world psychological or environmental reason."
        ),
        "hybrid_prompt": (
            "You are the Guild Architect & Royal Chronicler in a Village-Building RPG. "
            "Deliver concise GTNH 2.9.0 tactical coordinates and progression-safe advice wrapped in warm settlement-building lore (1-2 sentences)."
        ),
    },
    "Dark Fantasy RPG": {
        "slug": "Dark_Fantasy_RPG",
        "book_title": "The Ash & Iron Grimoire",
        "story_prompt": (
            "You are the Chronicler of 'The Ash & Iron Grimoire', a grim, atmospheric dark-fantasy novel. "
            "CRITICAL RULE: Never break the fourth wall. When the protagonist spins in circles, fights infernal beasts, requisitions 500 ore, "
            "or falls in battle and is resurrected by ancient soul-anchors, weave it into dramatic dark-fantasy prose (2-4 sentences) "
            "that invents a compelling in-world reason for their exact physical actions."
        ),
        "hybrid_prompt": (
            "You are the Veteran Warder & Arcane Chronicler in a Dark Fantasy RPG. "
            "Deliver exact GTNH 2.9.0 coordinates and tactical guidance framed as grim warder dispatches (1-2 sentences)."
        ),
    },
    "Crash-Landed Sovereign Artificer": {
        "slug": "Sovereign_Artificer",
        "book_title": "Logbook of the Fallen Star-Forge",
        "story_prompt": (
            "You are the shipboard AI Chronicler recording 'Logbook of the Fallen Star-Forge'. "
            "The protagonist is a stranded Sovereign Artificer rebuilding an interstellar forge from primitive bronze and steam. "
            "Turn spinning in circles (sensor recalibration / magnetic interference), ore requisitions, combat, and clone-bay resurrections "
            "into gripping sci-fi survival novel prose (2-4 sentences)."
        ),
        "hybrid_prompt": (
            "You are the Sovereign Shipboard AI & Tactical Artificer. "
            "Blend sci-fi survival atmosphere with razor-sharp GTNH 2.9.0 coordinates and engineering math (1-2 sentences)."
        ),
    },
    "Frontier Guildmaster's Expedition": {
        "slug": "Guild_Expedition",
        "book_title": "The Cartographer's Sovereign Ledger",
        "story_prompt": (
            "You are the Expedition Chronicler writing 'The Cartographer's Sovereign Ledger'. "
            "Turn every physical action—pacing in circles to take compass sightings, drafting 500-ore mining charters, "
            "fending off ambushes, or surviving near-fatal falls—into an epic 19th-century explorer's novel (2-4 sentences)."
        ),
        "hybrid_prompt": (
            "You are the Expedition Quartermaster & Master Cartographer. "
            "Provide exact GTNH 2.9.0 compass bearings and vein depths framed as official guild expedition dispatches (1-2 sentences)."
        ),
    },
}

PRODUCTION_DIRECTIVE = (
    "ACTIVE ENGINE MODE: PRODUCTION (CRITICAL-PATH SPEEDRUN & EFFICIENCY ENFORCER). "
    "Your prime directive is completing GregTech: New Horizons 2.9.0 with zero wasted movement. "
    "Call out any inefficiency or wrong-way movement immediately. Keep responses to 1-2 blunt, numbers-first sentences."
)


class StoryChronicler:
    """Tracks micro-behaviors (spinning, combat, death/resurrection, task requisitions) and writes a real-time novel."""

    def __init__(self):
        os.makedirs(CHRONICLES_DIR, exist_ok=True)
        self.mode = "PRODUCTION"
        self.theme_name = "Solo Village Builder & Homestead"
        self.chapter_num = 1
        self.entry_num = 0
        self.story_so_far: list[str] = []

        self._last_auto_ts = 0.0
        self._last_spin_ts = 0.0
        self._last_combat_story_ts = 0.0
        self._last_death_ts = 0.0

        self._seen_biomes: set[str] = set()
        self._seen_vein_chunks: set[tuple[int, int]] = set()
        self._known_death_waypoints: set[str] = set()
        self._waypoints_initialized = False

        # Rolling kinematic history: deque of (timestamp, x, y, z, crosshair_target, health)
        self._motion_window = deque(maxlen=28)
        self._was_dead_or_critical = False
        self._last_pos: tuple[float, float, float] | None = None
        self._last_health = 20.0

        self._load_state()

    def _load_state(self):
        if os.path.isfile(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.mode = data.get("mode", self.mode)
                self.theme_name = data.get("theme_name", self.theme_name)
                self.chapter_num = int(data.get("chapter_num", 1))
                self.entry_num = int(data.get("entry_num", 0))
                self.story_so_far = list(data.get("story_so_far", []))[-12:]
                self._seen_biomes = set(data.get("seen_biomes", []))
            except Exception:
                pass
        self._ensure_book_header()

    def save_state(self):
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mode": self.mode,
                        "theme_name": self.theme_name,
                        "chapter_num": self.chapter_num,
                        "entry_num": self.entry_num,
                        "story_so_far": self.story_so_far[-12:],
                        "seen_biomes": sorted(self._seen_biomes),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass

    def get_book_filepath(self) -> str:
        spec = STORY_THEMES.get(self.theme_name) or STORY_THEMES["Solo Village Builder & Homestead"]
        return os.path.join(CHRONICLES_DIR, f"Chronicle_{spec['slug']}.md")

    def _ensure_book_header(self):
        fp = self.get_book_filepath()
        if not os.path.isfile(fp):
            spec = STORY_THEMES.get(self.theme_name) or STORY_THEMES["Solo Village Builder & Homestead"]
            header = (
                f"# 📖 {spec['book_title']}\n"
                f"**Theme:** {self.theme_name}  \n"
                f"**Chronicle Begun:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
                f"**World Engine:** AA Dynamics Sovereign Gaming Engine (GTNH 2.9.0)\n\n"
                "---\n\n"
            )
            with open(fp, "w", encoding="utf-8") as f:
                f.write(header)

    def set_theme(self, theme_name: str):
        if theme_name in STORY_THEMES and theme_name != self.theme_name:
            self.theme_name = theme_name
            self._ensure_book_header()
            self.save_state()

    def set_mode(self, mode: str):
        mode = mode.upper().strip()
        if mode in ("PRODUCTION", "STORY", "HYBRID"):
            self.mode = mode
            self.save_state()

    def build_system_prompt(self, base_specialist_prompt: str, is_background_book: bool = False) -> tuple[str, float]:
        spec = STORY_THEMES.get(self.theme_name) or STORY_THEMES["Solo Village Builder & Homestead"]
        memory_block = ""
        if self.story_so_far:
            recent = "\n".join(f"- {s}" for s in self.story_so_far[-5:])
            memory_block = f"\n\n=== CHRONICLE MEMORY (THE STORY SO FAR) ===\n{recent}"

        if is_background_book or self.mode == "STORY":
            return f"{spec['story_prompt']}{memory_block}", 0.68
        if self.mode == "PRODUCTION":
            return f"{base_specialist_prompt}\n\n{PRODUCTION_DIRECTIVE}", 0.30
        return f"{base_specialist_prompt}\n\n{spec['hybrid_prompt']}{memory_block}", 0.50

    def build_task_requisition_story_prompt(self, task_title: str, subtasks: list[str], telemetry: dict[str, Any]) -> tuple[str, str]:
        """Builds a behind-the-scenes novel prompt when the player adds a task like 'Mine 500 copper ore'."""
        biome = telemetry.get("biome", "the frontier")
        bx, by, bz = int(telemetry.get("x", 0)), int(telemetry.get("y", 64)), int(telemetry.get("z", 0))
        held = telemetry.get("held_item", "surveyor's gear")
        sub_str = "; ".join(subtasks[:3]) if subtasks else "gathering raw materials"

        trigger_title = f"The Charter for {task_title}"
        prompt = (
            f"BEHIND-THE-SCENES STORY BEAT ({trigger_title}): While standing in {biome} at ({bx}, {by}, {bz}) holding {held}, "
            f"the protagonist just committed to a major new undertaking: '{task_title}' (requiring {sub_str}). "
            f"Write a rich, in-universe narrative passage (3-4 sentences) describing *why* they urgently need this quota "
            f"(e.g., drafting a guild requisition, inspecting half-built workshop frames that hunger for copper and tin, or bargaining with the frontier itself) "
            f"without ever mentioning 'voice commands' or 'task lists'."
        )
        return trigger_title, prompt

    def append_book_entry(self, event_trigger: str, narrative_prose: str, telemetry: dict[str, Any]) -> str:
        self._ensure_book_header()
        self.entry_num += 1
        if self.entry_num > 1 and (self.entry_num - 1) % 5 == 0:
            self.chapter_num += 1

        clean_prose = re.sub(r"§[0-9a-fk-or]", "", narrative_prose).strip()
        bx, by, bz = int(telemetry.get("x", 0)), int(telemetry.get("y", 64)), int(telemetry.get("z", 0))
        biome = telemetry.get("biome", "Unknown")
        dim = telemetry.get("dimension", "Overworld (0)")
        held = telemetry.get("held_item", "Empty Hand")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry_md = (
            f"### Chapter {self.chapter_num}, Entry {self.entry_num} — *{event_trigger}*\n"
            f"> **Location:** `({bx}, {by}, {bz})` | **Biome:** *{biome}* (`{dim}`) | **Equipped:** *{held}* | **Logged:** `{ts}`\n\n"
            f"{clean_prose}\n\n"
            "---\n\n"
        )
        fp = self.get_book_filepath()
        with open(fp, "a", encoding="utf-8") as f:
            f.write(entry_md)

        first_sentence = re.split(r"(?<=[.!?])\s+", clean_prose)[0]
        self.story_so_far.append(f"[Ch.{self.chapter_num}.{self.entry_num} @ {biome} ({bx},{bz})]: {first_sentence}")
        self.story_so_far = self.story_so_far[-12:]
        self.save_state()
        return fp

    def detect_live_player_behavior(self, telemetry: dict[str, Any], ore: dict[str, Any]) -> dict[str, str] | None:
        """
        Analyzes real-time kinematics, combat, death/resurrection, and exploration.
        Returns a dict {'kind': ..., 'title': ..., 'story_prompt': ..., 'prod_callout': ...} when a behavior triggers.
        """
        now = time.time()
        bx = float(telemetry.get("x", 0.0))
        by = float(telemetry.get("y", 64.0))
        bz = float(telemetry.get("z", 0.0))
        hp = float(telemetry.get("health", 20.0))
        biome = str(telemetry.get("biome") or "Unknown")
        held = str(telemetry.get("held_item") or "bare hands")
        target = str(telemetry.get("crosshair_target") or "the terrain")

        # --- 1. DEATH & RESURRECTION DETECTION ---
        summary_wp = (telemetry.get("discovered_veins_summary") or {}).get("nearest_waypoints", [])
        death_wps = {w.get("name", "") for w in summary_wp if "death" in str(w.get("name", "")).lower()}
        if not self._waypoints_initialized:
            self._known_death_waypoints = set(death_wps)
            self._waypoints_initialized = True
        new_death_wps = death_wps - self._known_death_waypoints
        self._known_death_waypoints.update(death_wps)

        sudden_respawn_jump = False
        if self._last_pos is not None:
            jump_dist = math.hypot(bx - self._last_pos[0], bz - self._last_pos[2])
            if jump_dist > 75.0 and (self._last_health <= 8.0 or self._was_dead_or_critical):
                sudden_respawn_jump = True

        if hp <= 0.5:
            self._was_dead_or_critical = True

        if (new_death_wps or sudden_respawn_jump or (self._was_dead_or_critical and hp >= 18.0)) and (now - self._last_death_ts) > 45.0:
            self._last_death_ts = now
            self._was_dead_or_critical = False
            self._last_pos = (bx, by, bz)
            self._last_health = hp
            death_marker = next(iter(new_death_wps), f"({int(bx)}, {int(by)}, {int(bz)})")
            return {
                "kind": "DEATH_RESURRECTION",
                "title": f"Fall and Resurrection at {biome}",
                "story_prompt": (
                    f"DRAMATIC DEATH & RESURRECTION EVENT: The protagonist was struck down in {biome} (leaving a fallen marker '{death_marker}') "
                    f"and has just awakened/resurrected at ({int(bx)}, {int(by)}, {int(bz)}) clutching {held}. "
                    f"Write a gripping 3-4 sentence passage narrating how they fell, the eerie threshold between life and death, "
                    f"and the supernatural, alchemical, or sheer iron-willed resurrection that pulled them back into the world."
                ),
                "prod_callout": f"Respawn detected at X {int(bx)}, Z {int(bz)}. Recover your gear at the death marker and get back on the active task!",
            }

        # --- 2. HEAVY COMBAT SKIRMISH STORY BEAT ---
        hp_drop = self._last_health - hp
        self._last_health = hp
        self._last_pos = (bx, by, bz)
        if (hp_drop >= 4.0 or (hp <= 9.0 and hp_drop >= 2.0)) and (now - self._last_combat_story_ts) > 65.0:
            self._last_combat_story_ts = now
            return {
                "kind": "COMBAT_SKIRMISH",
                "title": f"Bloodshed in {biome}",
                "story_prompt": (
                    f"LIVE COMBAT SCENE: While moving through {biome} at ({int(bx)}, {int(by)}, {int(bz)}), the protagonist took a vicious hit "
                    f"(dropping to {round(hp/2.0, 1)} hearts) while wielding {held} and facing {target}. "
                    f"Write a tense 3-sentence combat passage describing the ambush, the clash of {held}, and their desperate footwork to survive."
                ),
                "prod_callout": "",
            }

        # --- 3. SPINNING IN CIRCLES / PACING DETECTION ---
        self._motion_window.append((now, bx, by, bz, target, hp))
        recent_pts = [p for p in self._motion_window if (now - p[0]) <= 10.0]
        if len(recent_pts) >= 7 and (now - self._last_spin_ts) > 75.0:
            path_len = 0.0
            for i in range(1, len(recent_pts)):
                path_len += math.hypot(recent_pts[i][1] - recent_pts[i - 1][1], recent_pts[i][3] - recent_pts[i - 1][3])
            net_disp = math.hypot(recent_pts[-1][1] - recent_pts[0][1], recent_pts[-1][3] - recent_pts[0][3])
            distinct_targets = len({p[4] for p in recent_pts if p[4] and p[4] != "Air"})

            # Trigger if player walked >= 14 blocks in 10s while ending up within 5.5 blocks of start (circle/pacing)
            # OR swept crosshair across >= 5 distinct targets while staying within a 3.5-block radius
            is_circling = (path_len >= 14.0 and net_disp <= 5.5) or (distinct_targets >= 5 and net_disp <= 3.5 and path_len >= 4.0)
            if is_circling:
                self._last_spin_ts = now
                self._motion_window.clear()
                return {
                    "kind": "SPINNING_IN_CIRCLES",
                    "title": f"Whirling Bearings in {biome}",
                    "story_prompt": (
                        f"LIVE PLAYER ACTION — SPINNING IN CIRCLES: At ({int(bx)}, {int(by)}, {int(bz)}) in {biome}, "
                        f"the protagonist has been turning and pacing in tight circles while holding {held} and scanning past {target}. "
                        f"Invent a vivid, atmospheric in-story reason for why they are spinning in circles right now "
                        f"(e.g., searching the shadows for a stalking beast, disoriented by subterranean magnetic anomalies, "
                        f"testing the balance of {held}, or tracing an arcane perimeter) in 2-3 sentences."
                    ),
                    "prod_callout": f"Inefficiency alert: you're spinning in circles at X {int(bx)}, Z {int(bz)}. Lock onto your active task bearing and move!",
                }

        # --- 4. NEW BIOME OR 3x3 VEIN DISCOVERY (5-minute quiet cadence) ---
        if self.mode in ("STORY", "HYBRID") and (now - self._last_auto_ts) >= 240.0:
            if biome != "Unknown" and biome not in self._seen_biomes:
                self._seen_biomes.add(biome)
                self._last_auto_ts = now
                self.save_state()
                return {
                    "kind": "BIOME_DISCOVERY",
                    "title": f"Crossing into {biome}",
                    "story_prompt": (
                        f"NEW REGION DISCOVERY: The protagonist crossed into {biome} at ({int(bx)}, {int(by)}, {int(bz)}) "
                        f"carrying {held} and sighting {target}. Narrate the shifting landscape and mood in 2-3 sentences."
                    ),
                    "prod_callout": "",
                }

            if ore.get("is_in_ore_chunk"):
                ck = (int(ore.get("target_chunk_x", 0)), int(ore.get("target_chunk_z", 0)))
                if ck not in self._seen_vein_chunks:
                    self._seen_vein_chunks.add(ck)
                    self._last_auto_ts = now
                    return {
                        "kind": "VEIN_DISCOVERY",
                        "title": f"The Deep Vein Beneath {biome}",
                        "story_prompt": (
                            f"MINERAL STRIKE: Standing at chunk {ck} in {biome} with {held}, the protagonist's instruments "
                            f"locked onto a rich subterranean mineral vein. Narrate the discovery in 2-3 sentences."
                        ),
                        "prod_callout": "",
                    }

        return None